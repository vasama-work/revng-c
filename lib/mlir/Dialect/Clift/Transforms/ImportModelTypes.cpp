//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "revng-c/Tools/clift-opt/Model.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng-c/mlir/Dialect/Clift/IR/ImportModel.h"
#include "revng-c/mlir/Dialect/Clift/Transforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTIMPORTMODELTYPES
#include "revng-c/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace {

struct ImportModelTypesPass
  : mlir::clift::impl::CliftImportModelTypesBase<ImportModelTypesPass> {

  void runOnOperation() override {
    auto Module = getOperation();
    mlir::MLIRContext *const Context = Module->getContext();

    const auto EmitError = [&]() -> mlir::InFlightDiagnostic {
      return Context->getDiagEngine().emit(mlir::UnknownLoc::get(Context),
                                           mlir::DiagnosticSeverity::Error);
    };

    mlir::OpBuilder Builder(Module.getRegion());
    for (const auto &ModelType : Model->Types()) {
      auto CliftType = revng::getUnqualifiedTypeChecked(EmitError,
                                                        *Context,
                                                        *ModelType);

      Builder.create<mlir::clift::UndefOp>(mlir::UnknownLoc::get(Context),
                                           CliftType);
    }
  };
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::clift::createImportModelTypesPass() {
  return std::make_unique<ImportModelTypesPass>();
}
