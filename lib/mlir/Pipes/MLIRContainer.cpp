#include "revng-c/mlir/Pipes/MLIRContainer.h"
#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"

#include "revng-c/Pipes/Kinds.h"

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"

using namespace revng;
using namespace revng::pipes;

MLIRContainer::MLIRContainer()
  : pipeline::Container<MLIRContainer>("MLIRContainer")
{
  Context.loadDialect<mlir::clift::CliftDialect>();
}

std::unique_ptr<pipeline::ContainerBase>
MLIRContainer::cloneFiltered(const pipeline::TargetsList &Targets) const
{
  revng_abort();
}

llvm::Error MLIRContainer::extractOne(llvm::raw_ostream &OS,
                             const pipeline::Target &Target) const
{
  revng_abort();
}

pipeline::TargetsList MLIRContainer::enumerate() const
{
  const auto &Kind = kinds::MLIRFunctionKind;

  pipeline::TargetsList::List Targets;
  for (const mlir::Operation& O : Module->getRegion(0).front().getOperations()) {
    if (isa<mlir::FunctionOpInterface>(O)) {
      auto Symbol = cast<mlir::SymbolOpInterface>(O);
      Targets.emplace_back(Symbol.getNameAttr().getValue(), Kind);
    }
  }
  llvm::sort(Targets);

  return pipeline::TargetsList(std::move(Targets));
}

bool MLIRContainer::remove(const pipeline::TargetsList &Targets)
{
  const auto &Kind = kinds::MLIRFunctionKind;

  bool Result = false;
  for (const mlir::Operation& O : Module->getRegion(0).front().getOperations()) {
    auto Symbol = cast<mlir::SymbolOpInterface>(O);
    if (Targets.contains(pipeline::Target(Symbol.getNameAttr().getValue(), Kind))) {
      O.erase();
    }
  }
  return Result;
}

llvm::Error MLIRContainer::serialize(llvm::raw_ostream &OS) const
{
  Module->print(OS);
  return llvm::Error::success();
}

llvm::Error MLIRContainer::deserialize(const llvm::MemoryBuffer &Buffer)
{
  Module = mlir::parseSourceString<mlir::ModuleOp>(Buffer.getBuffer(), mlir::ParserConfig(&Context)).release();
  return llvm::Error::success();
}

std::vector<pipeline::Kind *> MLIRContainer::possibleKinds()
{
  return { &kinds::MLIRFunctionKind };
}
