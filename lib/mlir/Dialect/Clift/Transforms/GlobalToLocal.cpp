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
    mlir::Operation *Module = getOperation();

    struct GlobalUser {
      clift::FunctionOp Function;
      llvm::SmallVector<clift::UseOp> Users;

      explicit GlobalUser(clift::FunctionOp Function)
        : Function(Function) {}
    };

    // Globals used only in a single function are left with a valid FunctionOp.
    llvm::DenseMap<clift::GlobalVariableOp, GlobalUser> GlobalUsers;

    Module->walk([&](clift::FunctionOp F) {
      F->walk([&](clift::UseOp Use) {
        auto Symbol =
          mlir::SymbolTable::lookupSymbolIn(Module, Use.getSymbolNameAttr());

        if (auto Global = mlir::dyn_cast<clift::GlobalVariableOp>(Symbol)) {
          auto [Iterator, Inserted] = GlobalUsers.try_emplace(Global, F);

          if (not Inserted and F != Iterator->second.Function)
            Iterator->second.Function = {};
          else
            Iterator->second.Users.push_back(Use);
        }
      });
    });

    mlir::OpBuilder Builder(Module->getContext());
    for (auto &[Global, User] : GlobalUsers) {
      if (not User.Function)
        continue;

      Builder.setInsertionPointToStart(&User.Function.getBody().front());
      auto Local = Builder.create<clift::LocalVariableOp>(Global->getLoc(),
                                                          Global.getType(),
                                                          Global.getSymName());

      for (clift::UseOp Use : User.Users) {
        Use->replaceAllUsesWith(llvm::ArrayRef(Local.getResult()));
        Use->erase();
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<clift::ModuleOp>>
clift::createGlobalToLocalPass() {
  return std::make_unique<GlobalToLocalPass>();
}
