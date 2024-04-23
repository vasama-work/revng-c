//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "revng/Pipeline/RegisterContainerFactory.h"

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Pipes/MLIRContainer.h"

#include "MLIRLLVMHelpers.h"

using namespace revng;
using namespace revng::pipes;
using namespace revng::mlir_llvm;

using mlir::FunctionOpInterface;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::SymbolOpInterface;
using mlir::SymbolTable;

using ContextPtr = std::unique_ptr<MLIRContext>;
using OwningModuleRef = mlir::OwningOpRef<ModuleOp>;

static const mlir::DialectRegistry &getDialectRegistry() {
  static const mlir::DialectRegistry Registry = []() -> mlir::DialectRegistry {
    mlir::DialectRegistry Registry;
    // The DLTI dialect is used to express the data layout.
    Registry.insert<mlir::DLTIDialect>();
    // All dialects that implement the LLVMImportDialectInterface.
    mlir::registerAllFromLLVMIRTranslations(Registry);
    Registry.insert<mlir::clift::CliftDialect>();
    return Registry;
  }();
  return Registry;
}

static ContextPtr makeContext() {
  const auto Threading = MLIRContext::Threading::DISABLED;
  return std::make_unique<MLIRContext>(getDialectRegistry(), Threading);
}

// Cloning MLIR from one module into another requires first serialising the
// source and then deserialising it again into the target module. We do this
// a) when cloning operations from one container to another, and
// b) when "garbage collecting" types and attributes during remove.
//
// Ideally at least for use case a) we could clone from one context into another
// directly, but that is not supported at this time.
static OwningModuleRef cloneModuleInto(ModuleOp SourceModule,
                                       MLIRContext &DestinationContext) {
  llvm::SmallString<1024> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  mlir::writeBytecodeToFile(SourceModule, Out);

  // Serialised MLIR contains multiple operations and as such must be
  // deserialised into a block. We use a temporary block stored on stack.
  mlir::Block OuterBlock;

  const mlir::LogicalResult
    R = mlir::readBytecodeFile(llvm::MemoryBufferRef(Buffer, "cloneModuleInto"),
                               &OuterBlock,
                               mlir::ParserConfig(&DestinationContext));
  revng_assert(mlir::succeeded(R));

  // The serialised code is known to contain exactly one ModuleOp operation.
  auto &Operations = OuterBlock.getOperations();
  revng_assert(Operations.size() == 1);
  auto DestinationModule = mlir::cast<ModuleOp>(Operations.front());

#ifndef NDEBUG
  const size_t SrcOpsCount = getModuleOperations(SourceModule).size();
  const size_t NewOpsCount = getModuleOperations(DestinationModule).size();
  revng_assert(NewOpsCount == SrcOpsCount);
#endif

  // Detach the module from the temporary block before returning it.
  DestinationModule->remove();

  // The module is now owned by the OwningModuleRef returned to the caller.
  return DestinationModule;
}

static pipeline::Target makeTarget(const MetaAddress &MA) {
  return pipeline::Target(MA.toString(), kinds::MLIRLLVMFunctionKind);
}

static void makeExternal(FunctionOpInterface F) {
  revng_assert(F->getNumRegions() == 1);

  // A function is made external by clearing its region.
  mlir::Region &Region = F->getRegion(0);
  Region.dropAllReferences();
  Region.getBlocks().clear();
}

// Remove any unused non-target symbols.
static void pruneUnusedSymbols(ModuleOp Module) {
  llvm::DenseSet<Operation *> UsedSymbols;

  visit(Module, [&](FunctionOpInterface F) {
    if (isTargetFunction(F))
      UsedSymbols.insert(F);

    if (F.isExternal())
      return;

    if (const auto &Uses = SymbolTable::getSymbolUses(F)) {
      for (const SymbolTable::SymbolUse &Use : *Uses) {
        const auto &SymbolRef = Use.getSymbolRef();
        revng_assert(SymbolRef.getNestedReferences().empty());

        Operation *const
          Symbol = SymbolTable::lookupSymbolIn(Module,
                                               SymbolRef.getRootReference());
        revng_assert(Symbol);

        UsedSymbols.insert(Symbol);
      }
    }
  });

  visit(Module, [&](SymbolOpInterface S) {
    if (not UsedSymbols.contains(S.getOperation()))
      S->erase();
  });
}

const char MLIRContainer::ID = 0;

void MLIRContainer::setModule(OwningModuleRef &&NewModule) {
  revng_assert(NewModule);

  // Make any non-target functions external.
  visit(*NewModule, [&](FunctionOpInterface F) {
    if (not F.isExternal() and not isTargetFunction(F))
      makeExternal(F);
  });

  pruneUnusedSymbols(*NewModule);
  Module = std::move(NewModule);
}

// 1. Temporarily clone the source module within the source context.
// 2. Filter from the temporary module operations that we are not interested in.
// 3. Clone the temporary module into the context of the new container.
//
// Filtering in the source module and not after cloning into the new context
// is important to avoid polluting the new context with types and attributes
// not used by the remaining functions.
std::unique_ptr<pipeline::ContainerBase>
MLIRContainer::cloneFiltered(const pipeline::TargetsList &Filter) const {
  auto DestinationContainer = std::make_unique<MLIRContainer>(name());

  if (getModuleBlock(*Module).empty())
    return DestinationContainer;

  // The temporary module is automatically deleted at end of scope.
  OwningModuleRef TemporaryModule(mlir::cast<ModuleOp>((*Module)->clone()));

  bool RemovedSome = false;
  visit(*TemporaryModule, [&](FunctionOpInterface F) {
    if (F.isExternal())
      return;

    if (const auto MaybeMetaAddress = getMetaAddress(F)) {
      if (not Filter.contains(makeTarget(*MaybeMetaAddress))) {
        makeExternal(F);
        RemovedSome = true;
      }
    }
  });

  if (RemovedSome)
    pruneUnusedSymbols(*TemporaryModule);

  MLIRContext &DestinationContext = *DestinationContainer->Context;
  DestinationContext.appendDialectRegistry(Context->getDialectRegistry());

  OwningModuleRef &DestinationModule = DestinationContainer->Module;
  DestinationModule = cloneModuleInto(*TemporaryModule,
                                      *DestinationContainer->Context);

  return DestinationContainer;
}

void MLIRContainer::mergeBackImpl(MLIRContainer &&SourceContainer) {
  if (getModuleBlock(*SourceContainer.Module).empty())
    return;

  if (getModuleBlock(*Module).empty()) {
    Module = std::move(SourceContainer.Module);
    Context = std::move(SourceContainer.Context);
    return;
  }

  // Register the dialects of the other container in this container.
  Context->appendDialectRegistry(SourceContainer.Context->getDialectRegistry());

  // Clone the other container's module into this container's context.
  // This module is automatically erased at the end of scope.
  OwningModuleRef TemporaryModule = cloneModuleInto(*SourceContainer.Module,
                                                    *Context);

  mlir::Block &DestinationBlock = getModuleBlock(*Module);
  visit(*TemporaryModule, [&](SymbolOpInterface Symbol) {
    // Erase an existing symbol with the same name, if one exists.
    if (auto S = SymbolTable::lookupSymbolIn(*Module, Symbol.getName()))
      S->erase();

    // Move each new symbol from the temporary module to the container's module.
    Symbol->remove();
    DestinationBlock.push_back(Symbol);
  });

  // Assume that at least some symbols were copied over and always prune.
  pruneUnusedSymbols(*Module);
}

pipeline::TargetsList MLIRContainer::enumerate() const {
  pipeline::TargetsList::List List;

  visit(Module.get(), [&](FunctionOpInterface F) {
    if (F.isExternal())
      return;

    if (const auto MaybeMetaAddress = getMetaAddress(F))
      List.push_back(makeTarget(*MaybeMetaAddress));
  });

  // TargetsList requires ordering but does not itself sort the list.
  llvm::sort(List);

  return pipeline::TargetsList(std::move(List));
}

bool MLIRContainer::remove(const pipeline::TargetsList &List) {
  if (getModuleBlock(*Module).empty())
    return false;

  bool RemovedSome = false;
  visit(*Module, [&](FunctionOpInterface F) {
    if (F.isExternal())
      return;

    if (not isTargetFunction(F))
      return;

    makeExternal(F);
    RemovedSome = true;
  });

  if (RemovedSome) {
    // If any functions were removed, prune symbols and garbage collect types by
    // cloning the module into a new context.

    pruneUnusedSymbols(*Module);

    auto NewContext = makeContext();
    auto NewModule = cloneModuleInto(*Module, *NewContext);

    Module = std::move(NewModule);
    Context = std::move(NewContext);
  }

  return RemovedSome;
}

void MLIRContainer::clear() {
  auto NewContext = makeContext();

  Module = ModuleOp::create(mlir::UnknownLoc::get(NewContext.get()));
  Context = std::move(NewContext);
}

llvm::Error MLIRContainer::serialize(llvm::raw_ostream &OS) const {
  mlir::writeBytecodeToFile(*Module, OS);
  return llvm::Error::success();
}

llvm::Error MLIRContainer::deserialize(const llvm::MemoryBuffer &Buffer) {
  auto NewContext = makeContext();

  const mlir::ParserConfig Config(NewContext.get());
  OwningModuleRef
    NewModule = mlir::parseSourceString<ModuleOp>(Buffer.getBuffer(), Config);

  if (not NewModule)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot load MLIR module.");

  Module = std::move(NewModule);
  Context = std::move(NewContext);

  return llvm::Error::success();
}

llvm::Error MLIRContainer::extractOne(llvm::raw_ostream &OS,
                                      const pipeline::Target &Target) const {
  return cloneFiltered(pipeline::TargetsList::List{ Target })->serialize(OS);
}

static pipeline::RegisterDefaultConstructibleContainer<MLIRContainer> X;
