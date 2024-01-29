//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

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

  auto const &ModelBuffer = (*Buffer)->getBuffer();
  auto MaybeModel = TupleTree<model::Binary>::deserialize(ModelBuffer);
  if (not MaybeModel or not MaybeModel->verify()) {
    llvm::errs() << "Invalid Model\n";
    return EXIT_FAILURE;
  }

  std::error_code EC;
  llvm::raw_fd_ostream Dump{ OutFile, EC };
  if (EC)
    revng_abort(EC.message().c_str());

  const model::Binary &Model = **MaybeModel;

  mlir::MLIRContext Context;
  Context.loadDialect<mlir::clift::CliftDialect>();

  for (const auto &T : Model.Types()) {
    revng::getUnqualifiedType(Context, *T).print(Dump);
    Dump << "\n";
  }

  Dump.flush();
  EC = Dump.error();
  if (EC)
    revng_abort(EC.message().c_str());

  return EXIT_SUCCESS;
}
