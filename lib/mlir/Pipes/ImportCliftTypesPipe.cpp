//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/IRHelpers.h"
#include "revng/Support/IRHelpers.h"
#include "revng/TupleTree/TupleTree.h"

#include "revng-c/Pipes/Kinds.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng-c/mlir/Dialect/Clift/IR/ImportModel.h"
#include "revng-c/mlir/Pipes/MLIRContainer.h"

#include "MLIRLLVMHelpers.h"

using namespace revng::mlir_llvm;

using mlir::LLVM::LLVMFuncOp;
using LLVMCallOp = mlir::LLVM::CallOp;

namespace {

class MetadataTraits {
  using Location = pipeline::Location<decltype(revng::ranks::Instruction)>;

public:
  using BasicBlock = mlir::Block *;
  using Value = mlir::Operation *;
  using Function = LLVMFuncOp;
  using CallInst = LLVMCallOp;
  using KeyType = void *;

  static KeyType getKey(LLVMFuncOp F) { return F.getOperation(); }

  static KeyType getKey(mlir::Block *B) { return B; }

  static LLVMFuncOp getFunction(mlir::Operation *const O) {
    auto F = O->getParentOfType<LLVMFuncOp>();
    revng_assert(F);
    return F;
  }

  static std::optional<Location> getLocation(mlir::Operation *const O) {
    auto MaybeLoc = mlir::dyn_cast_or_null<LocType>(O->getLoc());

    if (not MaybeLoc)
      return std::nullopt;

    return Location::fromString(MaybeLoc.getMetadata().getName().getValue());
  }

  static TupleTree<efa::FunctionMetadata>
  extractFunctionMetadata(LLVMFuncOp F) {
    auto Attr = F->getAttrOfType<mlir::StringAttr>(FunctionMetadataMDName);
    const llvm::StringRef YAMLString = Attr.getValue();
    auto
      MaybeParsed = TupleTree<efa::FunctionMetadata>::deserialize(YAMLString);
    revng_assert(MaybeParsed and MaybeParsed->verify());
    return std::move(MaybeParsed.get());
  }

  static const model::Function *getModelFunction(const model::Binary &Binary,
                                                 LLVMFuncOp F) {
    auto const FI = mlir::cast<mlir::FunctionOpInterface>(F.getOperation());
    auto const MaybeFunctionName = getFunctionName(FI);

    if (not MaybeFunctionName)
      return nullptr;

    const auto MA = MetaAddress::fromString(*MaybeFunctionName);
    const auto It = Binary.Functions().find(MA);

    if (It == Binary.Functions().end())
      return nullptr;

    return &*It;
  }
};
using MLIRFunctionMetadataCache = BasicFunctionMetadataCache<MetadataTraits>;

class ImportCliftTypesPipe {
public:
  static constexpr auto Name = "import-clift-types";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(MLIRLLVMFunctionKind,
                                      0,
                                      MLIRLLVMFunctionKind,
                                      0,
                                      InputPreservation::Preserve) }) };
  }

  void run(const pipeline::ExecutionContext &Ctx,
           revng::pipes::MLIRContainer &MLIRContainer) {
    const TupleTree<model::Binary> &Model = revng::getModelFromContext(Ctx);

    mlir::ModuleOp Module = MLIRContainer.getModule();
    mlir::MLIRContext &Context = *Module.getContext();

    llvm::DenseSet<const model::Type *> ImportedTypes;

    mlir::OwningOpRef<mlir::ModuleOp>
      NewModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&Context));
    mlir::OpBuilder Builder(NewModule->getRegion());

    const auto importFunctionPrototype = [&](const model::Type &ModelType) {
      if (not ImportedTypes.insert(&ModelType).second)
        return;

      if (ImportedTypes.size() == 1)
        Context.loadDialect<mlir::clift::CliftDialect>();

      const auto EmitError = [&]() -> mlir::InFlightDiagnostic {
        return Context.getDiagEngine().emit(mlir::UnknownLoc::get(&Context),
                                            mlir::DiagnosticSeverity::Error);
      };

      const auto CliftType = revng::getUnqualifiedTypeChecked(EmitError,
                                                              Context,
                                                              ModelType);
      revng_assert(CliftType);

      Builder.create<mlir::clift::UndefOp>(mlir::UnknownLoc::get(&Context),
                                           CliftType);
    };

    MLIRFunctionMetadataCache Cache;
    visit(Module, [&](mlir::FunctionOpInterface F) {
      const auto Name = getFunctionName(F);

      if (not Name)
        return;

      const auto &ModelFunctions = Model->Functions();
      const auto It = ModelFunctions.find(MetaAddress::fromString(*Name));
      revng_assert(It != ModelFunctions.end());
      const model::Function &ModelFunction = *It;

      importFunctionPrototype(*ModelFunction.Prototype().getConst());

      if (F.isExternal())
        return;

      F->walk([&](LLVMCallOp Call) {
        const auto CalleePrototype = Cache.getCallSitePrototype(*Model,
                                                                Call,
                                                                &ModelFunction);

        if (CalleePrototype.empty())
          return;

        importFunctionPrototype(*CalleePrototype.getConst());
      });
    });

    auto &OldBlock = getModuleBlock(Module);
    auto &NewBlock = getModuleBlock(*NewModule);

    while (not NewBlock.empty()) {
      mlir::Operation &NewOperation = NewBlock.front();
      NewOperation.remove();
      OldBlock.push_back(&NewOperation);
    }
  }
};

static pipeline::RegisterPipe<ImportCliftTypesPipe> X;
} // namespace
