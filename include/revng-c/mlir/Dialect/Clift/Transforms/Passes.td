//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_PASSES
#define MLIR_CLIFT_PASSES

include "mlir/Pass/PassBase.td"

def CliftGlobalToLocal : Pass<"global-to-local", "mlir::clift::ModuleOp"> {
  let summary = "Convert global variables to local variables when used only within a single function";
  let constructor = "mlir::clift::createGlobalToLocalPass()";
  let dependentDialects = ["clift::CliftDialect"];
}

#endif // MLIR_CLIFT_PASSES
