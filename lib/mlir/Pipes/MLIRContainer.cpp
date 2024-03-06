//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "revng/Pipeline/RegisterContainerFactory.h"

#include "revng-c/Pipes/Kinds.h"
#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Pipes/MLIRContainer.h"

#include <optional>

using namespace revng;
using namespace revng::pipes;

using mlir::LLVM::LLVMFuncOp;

static constexpr auto& TheKind = kinds::MLIRFunctionKind;

using ContextPtr = std::unique_ptr<mlir::MLIRContext>;
using OwningModuleRef = mlir::OwningOpRef<mlir::ModuleOp>;

static mlir::Block &getModuleBlock(mlir::ModuleOp Module) {
  revng_assert(Module);
  revng_assert(Module->getNumRegions() == 1);
  return Module->getRegion(0).front();
}

static decltype(auto) getModuleOperations(mlir::ModuleOp Module) {
  return getModuleBlock(Module).getOperations();
}

static OwningModuleRef cloneModule(mlir::MLIRContext& Context, mlir::ModuleOp Module) {
  llvm::SmallString<1024> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  mlir::writeBytecodeToFile(Module, Out);

  mlir::Block OuterBlock;
  mlir::LogicalResult R = mlir::readBytecodeFile(
    llvm::MemoryBufferRef(Buffer, "BYTECODE1"),
    &OuterBlock,
    mlir::ParserConfig(&Context));
  revng_assert(mlir::succeeded(R));

  auto &Operations = OuterBlock.getOperations();
  revng_assert(Operations.size() == 1);

  auto NewModule = mlir::cast<mlir::ModuleOp>(Operations.front());
  NewModule->remove();

  revng_assert(
    getModuleOperations(NewModule).size() ==
    getModuleOperations(Module).size());

  return NewModule;
}

static std::optional<llvm::StringRef> getLinkageName(mlir::Operation &O) {
  using LocType = mlir::FusedLocWith<mlir::LLVM::DISubprogramAttr>;
  if (const auto L = mlir::dyn_cast<LocType>(O.getLoc()))
    return L.getMetadata().getLinkageName();
  return std::nullopt;
}

static std::optional<llvm::StringRef> getFunctionName(mlir::Operation &O) {
  revng_assert(mlir::isa<mlir::FunctionOpInterface>(O));
  if (const auto Name = getLinkageName(O)) {
    static constexpr llvm::StringRef Path = "/function/";
    revng_assert(Name->starts_with(Path));
    return Name->substr(Path.size());
  }
  return std::nullopt;
}

static pipeline::Target makeTarget(const llvm::StringRef Name) {
  return pipeline::Target(Name, TheKind);
}

static llvm::StringRef getFunctionName(const pipeline::Target &Target) {
  revng_check(&Target.getKind() == &TheKind);
  revng_check(Target.getPathComponents().size() == 1);

  auto&& Name = Target.getPathComponents().front();
  static_assert(std::is_lvalue_reference_v<decltype(Name)>);
  return Name;
}

static bool isExternal(mlir::Operation *const O) {
  return mlir::cast<mlir::FunctionOpInterface>(O).isExternal();
}


const char MLIRContainer::ID = 0;

MLIRContainer::MLIRContainer(const llvm::StringRef Name) :
  pipeline::Container<MLIRContainer>(Name) {}

void MLIRContainer::setModule(mlir::OwningOpRef<mlir::ModuleOp> &&NewModule) {
  revng_assert(NewModule);
  revng_assert(NewModule->getContext() == Context.get());

  Targets.clear();
  for (mlir::Operation &O : getModuleOperations(*NewModule)) {
    if (not mlir::isa<LLVMFuncOp>(O))
      continue;

    if (const auto Name = getFunctionName(O)) {
      const auto R = Targets.try_emplace(*Name, &O);
      revng_assert(R.second);
    }
  }

  Module = std::move(NewModule);

  prune();
}

std::unique_ptr<pipeline::ContainerBase>
MLIRContainer::cloneFiltered(const pipeline::TargetsList &Filter) const {
  auto NewContainer = std::make_unique<MLIRContainer>(name());
  if (Context) {
    NewContainer->Context = makeContext();
    NewContainer->Context->appendDialectRegistry(Context->getDialectRegistry());
    NewContainer->Module = cloneModule(*NewContainer->Context, *Module);

    for (const auto &[Name, Operation] : Targets) {
      if (mlir::cast<LLVMFuncOp>(Operation).isExternal())
        continue;

      mlir::Operation *const NewOperation = mlir::SymbolTable::lookupSymbolIn(
        *NewContainer->Module,
        mlir::SymbolTable::getSymbolName(Operation).getValue());
      revng_assert(NewOperation);

      const auto R = NewContainer->Targets.try_emplace(Name, NewOperation);
      revng_assert(R.second);

      if (not Filter.contains(makeTarget(Name))) {
        // Clear the operation blocks to make it external.
        NewOperation->getRegion(0).getBlocks().clear();
      }
    }

    NewContainer->prune();
  }
  return NewContainer;
}

void MLIRContainer::mergeBackImpl(MLIRContainer &&Container) {
  if (not Container.Context)
    return;

  if (Context) {
    using Iterator = decltype(Targets)::iterator;
    llvm::SmallVector<std::pair<Iterator, mlir::Operation*>, 16> NewTargets;

    for (const auto &[Name, Operation] : Targets) {
      const auto R = Container.Targets.try_emplace(Name, nullptr);

      if (R.second || isExternal(R.first->second)) {
        NewTargets.emplace_back(R.first, Operation);
      } else {
        Operation->erase();
      }
    }

    if (not NewTargets.empty()) {
      prune();
      OwningModuleRef NewModule = cloneModule(*Container.Context, *Module);
      for (auto const &[NewIterator, Operation] : NewTargets) {
        mlir::Operation *const NewOperation = mlir::SymbolTable::lookupSymbolIn(
          *NewModule,
          mlir::SymbolTable::getSymbolName(Operation));
        revng_assert(NewOperation);

        if (NewIterator->second)
          NewIterator->second->erase();

        NewOperation->remove();
        getModuleBlock(*Container.Module).push_back(NewOperation);
      }
    }
  }

  Targets = std::move(Container.Targets);
  Module = std::move(Container.Module);
  Context = std::move(Container.Context);
}

pipeline::TargetsList MLIRContainer::enumerate() const {
  pipeline::TargetsList::List List;

  if (Context) {
    List.reserve(Targets.size());
    for (const auto &[Name, O] : Targets)
      List.push_back(makeTarget(Name));
    llvm::sort(List);
  }

  return pipeline::TargetsList(std::move(List));
}

bool MLIRContainer::remove(const pipeline::TargetsList &Targets) {
  bool Result = false;
  for (const pipeline::Target &T : Targets) {
    if (const auto I = this->Targets.find(getFunctionName(T)); I != this->Targets.end()) {
      mlir::Operation &O = *I->second;
      revng_assert(O.getNumRegions() == 1);
      auto &Blocks = O.getRegion(0).getBlocks();
      if (not Blocks.empty()) {
        Blocks.clear();
        Result = true;
      }
    }
  }
  return Result;
}

void MLIRContainer::clear() {
  Targets.clear();
  Module = {};
  Context = {};
}

llvm::Error MLIRContainer::serialize(llvm::raw_ostream &OS) const {
  if (Context)
    Module->print(OS, mlir::OpPrintingFlags().enableDebugInfo());
  return llvm::Error::success();
}

llvm::Error MLIRContainer::deserialize(const llvm::MemoryBuffer &Buffer) {
  revng_assert(not Context);

  Context = makeContext();
  setModule(mlir::parseSourceString<mlir::ModuleOp>(Buffer.getBuffer(),
                                                   mlir::ParserConfig(Context.get())));

  // WIP: Decide return value
  return llvm::Error::success();
}

llvm::Error MLIRContainer::extractOne(llvm::raw_ostream &OS,
                                      const pipeline::Target &Target) const {
  if (const auto I = Targets.find(getFunctionName(Target)); I != Targets.end())
    mlir::writeBytecodeToFile(I->second, OS);
  // WIP
  return llvm::Error::success();
}

std::vector<pipeline::Kind *> MLIRContainer::possibleKinds() {
  return { &TheKind };
}

void MLIRContainer::prune() {
  llvm::DenseSet<mlir::Operation *> Symbols;

  for (mlir::Operation &O : getModuleOperations(*Module)) {
    Symbols.insert(&O);
  }

  for (const auto &[Name, Operation] : Targets) {
    Symbols.erase(Operation);

    if (isExternal(Operation))
      continue;

    if (const auto &Uses = mlir::SymbolTable::getSymbolUses(Operation)) {
      for (const mlir::SymbolTable::SymbolUse &Use : *Uses) {
        const auto &SymbolRef = Use.getSymbolRef();
        revng_assert(SymbolRef.getNestedReferences().empty());

        mlir::Operation *const Symbol = mlir::SymbolTable::lookupSymbolIn(
          *Module,
          SymbolRef.getRootReference());
        revng_assert(Symbol);

        Symbols.erase(Symbol);
      }
    }
  }

  for (mlir::Operation *const O : Symbols) {
    O->erase();
  }
}

std::unique_ptr<mlir::MLIRContext> MLIRContainer::makeContext() {
  return std::make_unique<mlir::MLIRContext>(mlir::MLIRContext::Threading::DISABLED);
}

static pipeline::RegisterDefaultConstructibleContainer<MLIRContainer> X;
