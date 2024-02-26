//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Transforms/Utils/Cloning.h"

#include "mlir/Target/LLVMIR/Import.h"

#include "revng/Pipeline/RegisterPipe.h"

#include "revng-c/Pipes/Kinds.h"
#include "revng-c/mlir/Pipes/MLIRContainer.h"

namespace {

class ImportLLVMToMLIRPipe {
public:
  static constexpr auto Name = "import-llvm-to-mlir2";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(StackAccessesSegregated,
                                      0,
                                      MLIRFunctionKind,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(const pipeline::ExecutionContext &Ctx,
           pipeline::LLVMContainer &IRContainer,
           revng::pipes::MLIRContainer &MLIRContainer) {

    // Let's do the MLIR import on a cloned Module, so we can save the old one
    // untouched.
    llvm::ValueToValueMapTy Map;
    auto ClonedModule = llvm::CloneModule(IRContainer.getModule(), Map);

    // Import LLVM Dialect.
    auto Module = translateLLVMIRToModule(
      std::move(ClonedModule),
      &MLIRContainer.getContext());

    revng_check(mlir::succeeded(Module->verify()));

    MLIRContainer.setModule(std::move(Module));
  }

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const {
    // WIP
    //OS << "mlir-translate -import-llvm -mlir-print-debuginfo module.ll -o "
    //      "module.mlir\n";
  }
};

static pipeline::RegisterPipe<ImportLLVMToMLIRPipe> X;
} // namespace
