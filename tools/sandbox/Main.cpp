//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "revng/Support/InitRevng.h"

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/IR/ImportModel.h"

using namespace llvm::cl;

static const char *Overview = "Standalone tool that ingests an instance of "
                              "revng's model and..."; // WIP

static OptionCategory Category("revng-clift-import-model options");

static opt<std::string> OutFile("o",
                                init("-" /* for stdout */),
                                desc("Output MLIR file"),
                                cat(Category));

static opt<std::string> InFile("i",
                               init("-" /* for stdin */),
                               desc("Input file with Model"),
                               cat(Category));

int main(int Argc, const char **Argv) {
  revng::InitRevng X(Argc, Argv, Overview, { &Category });

  const auto Buffer = llvm::MemoryBuffer::getFileOrSTDIN(InFile);
  if (std::error_code EC = Buffer.getError())
    revng_abort(EC.message().c_str());

  mlir::MLIRContext Context;
  mlir::DialectRegistry Registry;

  Registry.insert<mlir::DLTIDialect>();
  Registry.insert<mlir::LLVM::LLVMDialect>();

  Context.appendDialectRegistry(Registry);
  Context.loadAllAvailableDialects();

  auto Module = mlir::parseSourceString<mlir::ModuleOp>((*Buffer)->getBuffer(),
                                          mlir::ParserConfig(&Context));

  mlir::Operation *Symbol = mlir::SymbolTable::lookupSymbolIn(
    *Module, "local__function_0x400118_Code_x86_64");
  revng_assert(Symbol);

  Symbol->walk([&](mlir::Operation *O) {
    fprintf(stderr, "%s\n", std::string(O->getName().getIdentifier().getValue()).c_str());
    for (const mlir::OpOperand &Operand : O->getUses()) {
      mlir::Operation *Dependency = Operand.get().getDefiningOp();
      if (mlir::isa<mlir::SymbolOpInterface>(Dependency)) {
        fprintf(stderr, "  %s\n", std::string(mlir::SymbolTable::getSymbolName(Dependency).getValue()).c_str());
      }
    }
  });

  return EXIT_SUCCESS;
}
