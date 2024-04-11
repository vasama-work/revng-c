//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "revng/Support/Assert.h"
#include "revng/Support/InitRevng.h"

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"

using namespace llvm::cl;

static constexpr char ToolName[] = "Standalone optimizer driver\n";

int main(int Argc, char *Argv[]) {
  mlir::DialectRegistry Registry;

  Registry.insert<mlir::DLTIDialect>();
  Registry.insert<mlir::LLVM::LLVMDialect>();
  Registry.insert<mlir::clift::CliftDialect>();

  mlir::LLVM::registerLLVMPasses();

  using mlir::asMainReturnCode;
  using mlir::MlirOptMain;
  std::string ToolName = "Standalone optimizer driver\n";

  return asMainReturnCode(MlirOptMain(Argc, Argv, ToolName, Registry));

  return 0;
}
