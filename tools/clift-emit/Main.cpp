//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "revng/Support/Assert.h"
#include "revng/Support/InitRevng.h"

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/Transforms/ModelOption.h"
#include "revng-c/mlir/Dialect/Clift/Utils/CBackend.h"

namespace clift = mlir::clift;
namespace cl = llvm::cl;

using namespace clift;

int main(int Argc, char **Argv) {
  static cl::opt<std::string> InputFilename(cl::Positional,
                                            cl::desc("<input file>"),
                                            cl::init("-"));

  static cl::opt<clift::ModelOptionType> Model("model",
                                               cl::desc("Model file path"));

  static cl::opt<bool> Tagless("tagless", cl::init(false));
  static cl::opt<bool> EmitDeclaration("decl", cl::init(false));

  llvm::InitLLVM Init(Argc, Argv);

  cl::ParseCommandLineOptions(Argc, Argv, "clift-emit");

  std::string ErrorMessage;
  auto InputFile = mlir::openInputFile(InputFilename, &ErrorMessage);
  if (!InputFile) {
    llvm::errs() << ErrorMessage << "\n";
    return EXIT_FAILURE;
  }

  mlir::DialectRegistry Registry;
  Registry.insert<CliftDialect>();

  mlir::MLIRContext Context(mlir::MLIRContext::Threading::DISABLED);
  Context.appendDialectRegistry(Registry);
  Context.loadAllAvailableDialects();

  auto Module =
    mlir::parseSourceString<clift::ModuleOp>(InputFile->getBuffer(),
                                             mlir::ParserConfig(&Context));

  if (not Module)
    return EXIT_FAILURE;

  c_backend::PlatformInfo Platform = {
    .sizeof_char = 1,
    .sizeof_short = 2,
    .sizeof_int = 4,
    .sizeof_long = 8,
    .sizeof_longlong = 8,
    .sizeof_pointer = 8,
  };

  Module->walk([&](FunctionOp F) {
    printf("%s\n", c_backend::emit(Platform,
                                   *Model,
                                   F,
                                   Tagless,
                                   EmitDeclaration).c_str());
  });

  return EXIT_SUCCESS;
}
