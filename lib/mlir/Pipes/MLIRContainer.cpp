//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"

#include "revng/Pipeline/RegisterContainerFactory.h"

#include "revng-c/Pipes/Kinds.h"
#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Pipes/MLIRContainer.h"

using namespace revng;
using namespace revng::pipes;

namespace {

static constexpr auto& TheKind = kinds::MLIRFunctionKind;

using OwningModuleRef = mlir::OwningOpRef<mlir::ModuleOp>;

static mlir::MLIRContext makeContext() {
  return mlir::MLIRContext(mlir::MLIRContext::Threading::DISABLED);
}

static OwningModuleRef createModule(mlir::MLIRContext& Context) {
  return mlir::ModuleOp::create(mlir::UnknownLoc::get(&Context));
}

static mlir::Block &getModuleBlock(mlir::ModuleOp Module) {
  return Module->getRegion(0).front();
}

static decltype(auto) getModuleOperations(mlir::ModuleOp Module) {
  return getModuleBlock(Module).getOperations();
}

static OwningModuleRef cloneModule(mlir::MLIRContext& Context, mlir::ModuleOp Module) {
  llvm::SmallString<1024> Buffer;

  OwningModuleRef NewModule = createModule(Context);
  mlir::Block *NewModuleBlock = &getModuleBlock(NewModule.get());

  for (mlir::Operation &O : getModuleBlock(Module).getOperations()) {
    llvm::raw_svector_ostream Out(Buffer);
    mlir::writeBytecodeToFile(&O, Out);

    mlir::LogicalResult R = mlir::readBytecodeFile(
      llvm::MemoryBufferRef(Buffer, "BYTECODE"),
      NewModuleBlock,
      mlir::ParserConfig(&Context));
    revng_assert(R.succeeded());

    Buffer.clear();
  }

  return NewModule;
}

static pipeline::Target getTarget(mlir::Operation &O) {
  auto Symbol = mlir::cast<mlir::SymbolOpInterface>(O);
  return pipeline::Target(Symbol.getNameAttr().getValue(), TheKind);
}

static llvm::StringRef getSymbolName(const pipeline::Target &Target) {
  revng_check(&Target.getKind() == &TheKind);
  revng_check(Target.getPathComponents().size() == 1);

  auto&& Name = Target.getPathComponents().front();
  static_assert(std::is_lvalue_reference_v<decltype(Name)>);
  return Name;
}

static mlir::Operation *findSymbol(mlir::ModuleOp Module, const pipeline::Target &Target) {
  if (mlir::Operation *const O =
      mlir::SymbolTable::lookupSymbolIn(Module, getSymbolName(Target))) {
    revng_assert(mlir::isa<mlir::FunctionOpInterface>(*O));
    return O;
  }
  return nullptr;
}

class OperationCopier
{
  mlir::MLIRContext &TargetContext;
  mlir::Block &TargetBlock;
  llvm::SmallString<1024> Buffer;

public:
  explicit OperationCopier(mlir::MLIRContext &TargetContext, mlir::Block &TargetBlock)
    : TargetContext(TargetContext),
      TargetBlock(TargetBlock) {}

  void copy(mlir::Operation &SourceOperation) {
    llvm::raw_svector_ostream Out(Buffer);
    mlir::writeBytecodeToFile(&SourceOperation, Out);

    mlir::LogicalResult R = mlir::readBytecodeFile(
      llvm::MemoryBufferRef(Buffer, "BYTECODE"),
      &TargetBlock,
      mlir::ParserConfig(&TargetContext));
    revng_assert(mlir::succeeded(R));

    Buffer.clear();
  }
};

} // namespace

const char MLIRContainer::ID = 0;

MLIRContainer::MLIRContainer(const llvm::StringRef Name) :
  pipeline::Container<MLIRContainer>(Name),
  Context(makeContext()) {
  Context.loadDialect<mlir::clift::CliftDialect>();
}

std::unique_ptr<pipeline::ContainerBase>
MLIRContainer::cloneFiltered(const pipeline::TargetsList &Targets) const {
#if 1
  auto NewContainer = std::make_unique<MLIRContainer>(name());
  OperationCopier copier(NewContainer->Context, getModuleBlock(*NewContainer->Module));

  for (const pipeline::Target &T : Targets) {
    if (mlir::Operation *const O = findSymbol(*Module, T)) {
      copier.copy(*O);
    }
  }
#else
  OwningModuleRef NewModule = createModule(Context);
  mlir::Block &NewModuleBlock = getModuleBlock(*NewContainer->Module);

  for (mlir::Operation &O : getModuleOperations(*Module)) {
    if (mlir::isa<mlir::FunctionOpInterface>(O)) {
      auto Symbol = cast<mlir::SymbolOpInterface>(O);
      if (Targets.contains(pipeline::Target(Symbol.getNameAttr().getValue(), TheKind)))
        NewModuleBlock.push_back(O.clone());
    }
  }
  auto NewContainer = std::make_unique<MLIRContainer>(name());
  Container->Module = cloneModule(Container->Context, *NewModule).release();
#endif

  return NewContainer;
}

void MLIRContainer::mergeBackImpl(MLIRContainer &&Container) {
#if 1
  OwningModuleRef NewModule = cloneModule(Context, *Container.Module);
  mlir::Block &Block = getModuleBlock(*Module);

  for (mlir::Operation &NewOperation : getModuleOperations(*NewModule)) {
    if (mlir::Operation *const OldOperation = mlir::SymbolTable::lookupSymbolIn(
          Module.get(),
          cast<mlir::SymbolOpInterface>(NewOperation).getNameAttr()))
      OldOperation->erase();
    NewOperation.remove();
    Block.push_back(&NewOperation);
  }

#else
  OwningModuleRef SourceModule = createModule(Container.Context);
  for (mlir::Operation &O : getModuleOperations(*Container.Module)) {
    if (mlir::isa<mlir::FunctionOpInterface>(O)) {
      auto Symbol = cast<mlir::SymbolOpInterface>(O);
      if (not mlir::SymbolTable::lookupSymbolIn(Module.get(), Symbol.getNameAttr())) {
        O.remove();
        SourceModule->push_back(&O);
      }
    }
  }

  OwningModuleRef TargetModule = cloneModule(Context, *SourceModule);
  for (mlir::Operation &O : getModuleOperations(*TargetModule)) {
    if (mlir::isa<mlir::FunctionOpInterface>(O)) {
      O.remove();
      Module->push_back(&O);
    }
  }
#endif
}

pipeline::TargetsList MLIRContainer::enumerate() const {
  pipeline::TargetsList::List Targets;
  for (mlir::Operation &O : getModuleOperations(*Module)) {
    if (mlir::isa<mlir::FunctionOpInterface>(O)) {
      Targets.push_back(getTarget(O));
    }
  }
  llvm::sort(Targets);

  return pipeline::TargetsList(std::move(Targets));
}

bool MLIRContainer::remove(const pipeline::TargetsList &Targets) {
  bool Result = false;
  for (const pipeline::Target &T : Targets) {
    if (mlir::Operation *const O = findSymbol(*Module, T)) {
      O->erase();
      Result = true;
    }
  }
  return Result;
}

void MLIRContainer::clear() {
  getModuleBlock(*Module).clear();
}

llvm::Error MLIRContainer::serialize(llvm::raw_ostream &OS) const {
  mlir::writeBytecodeToFile(Module.get(), OS);
  return llvm::Error::success();
}

llvm::Error MLIRContainer::deserialize(const llvm::MemoryBuffer &Buffer) {
  Module = mlir::parseSourceString<mlir::ModuleOp>(Buffer.getBuffer(),
                                                   mlir::ParserConfig(&Context))
             .release();
  // WIP
  return llvm::Error::success();
}

llvm::Error MLIRContainer::extractOne(llvm::raw_ostream &OS,
                                      const pipeline::Target &Target) const {
  if (mlir::Operation *const O = findSymbol(*Module, Target)) {
    mlir::writeBytecodeToFile(O, OS);
  }
  // WIP
  return llvm::Error::success();
}

std::vector<pipeline::Kind *> MLIRContainer::possibleKinds() {
  return { &TheKind };
}

static pipeline::RegisterDefaultConstructibleContainer<MLIRContainer> X;
