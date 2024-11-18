//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng-c/mlir/Dialect/Clift/Transforms/Passes.h"
#include "revng-c/mlir/Dialect/Clift/Utils/ImportModel.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTGLOBALTOLOCAL
#include "revng-c/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

using clift::impl::CliftGlobalToLocalBase;
struct GlobalToLocalPass : CliftGlobalToLocalBase<GlobalToLocalPass> {
  void runOnOperation() override {
    auto Module = mlir::cast<clift::ModuleOp>(getOperation());

    auto findOnlyFunctionUser = [](const mlir::SymbolTable::UseRange &Uses) {
      clift::FunctionOp UserFunction = {};

      for (const mlir::SymbolTable::SymbolUse &Use : Uses) {
        auto Function = Use.getUser()->getParentOfType<clift::FunctionOp>();

        if (UserFunction and UserFunction != Function)
          return clift::FunctionOp{};

        UserFunction = Function;
      }

      return UserFunction;
    };

    mlir::OpBuilder Builder(Module->getContext());
    for (auto Global : Module.getBody().getOps<clift::GlobalVariableOp>()) {
      auto Uses = mlir::SymbolTable::getSymbolUses(Global.getOperation(),
                                                   Module.getOperation());

      // getSymbolUses returns nullopt if there are unknown operations which may
      // contain symbol uses. This is not the case for valid Clift ModuleOps.
      revng_assert(Uses.has_value());

      if (clift::FunctionOp Function = findOnlyFunctionUser(*Uses)) {
        Builder.setInsertionPointToStart(&Function.getBody().front());
        auto Local = Builder.create<clift::LocalVariableOp>(Global->getLoc(),
                                                            Global.getType(),
                                                            Global.getSymName());

        for (const mlir::SymbolTable::SymbolUse &Use : *Uses) {
          mlir::Operation *User = Use.getUser();
          User->replaceAllUsesWith(llvm::ArrayRef(Local.getResult()));
          User->erase();
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<clift::ModuleOp>>
clift::createGlobalToLocalPass() {
  return std::make_unique<GlobalToLocalPass>();
}
