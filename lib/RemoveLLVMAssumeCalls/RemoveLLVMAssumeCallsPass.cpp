//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/IsolatedFunctions/IsolatedFunctions.h"
#include "revng-c/RemoveLLVMAssumeCalls/RemoveLLVMAssumeCallsPass.h"

using namespace llvm;

using RemoveAssumePass = RemoveLLVMAssumeCallsPass;

char RemoveAssumePass::ID = 0;
using Reg = RegisterPass<RemoveAssumePass>;
static Reg
  X("remove-llvmassume-calls", "Removes calls to assume intrinsic", true, true);

void RemoveAssumePass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelPass>();
}

bool RemoveAssumePass::runOnFunction(Function &F) {

  // Skip non-isolated functions
  const model::Binary &Model = getAnalysis<LoadModelPass>().getReadOnlyModel();
  if (not hasIsolatedFunction(Model, F))
    return false;

  // Remove calls to `llvm.assume` in isolated functions.
  SmallVector<Instruction *, 8> ToErase;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB)
      if (auto *C = dyn_cast<CallInst>(&I))
        if (getCallee(C)->getName() == "llvm.assume")
          ToErase.push_back(C);
  }

  bool Changed = not ToErase.empty();
  for (Instruction *I : ToErase)
    I->eraseFromParent();

  return Changed;
}
