#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Transforms/Scalar.h>

#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

#include <revng/Support/IRHelpers.h>

#include "revng-c/RestructureCFGPass/ASTTree.h"

#include "revng-c/DecompilationPass.h"

#include "DecompilationAction.h"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

char DecompilationPass::ID = 0;

static RegisterPass<DecompilationPass>
  X("decompilation", "Decompilation Pass", false, false);

static cl::OptionCategory RevNgCategory("revng options");

DecompilationPass::DecompilationPass(std::unique_ptr<llvm::raw_ostream> Out) :
  llvm::FunctionPass(ID),
  Out(std::move(Out)) {
}

DecompilationPass::DecompilationPass() :
  llvm::FunctionPass(ID),
  Out(nullptr) {
}

static void processFunction(llvm::Function &F) {
  legacy::FunctionPassManager OptPM(F.getParent());
  OptPM.add(createSROAPass());
  OptPM.add(createConstantPropagationPass());
  OptPM.add(createDeadCodeEliminationPass());
  OptPM.add(createEarlyCSEPass());
  OptPM.run(F);
}

bool DecompilationPass::runOnFunction(llvm::Function &F) {
  if (not F.getName().startswith("bb."))
    return false;
  // HACK!!!
  if (F.getName().startswith("bb.quotearg_buffer_restyled")
      or F.getName().startswith("bb._getopt_internal_r")
      or F.getName().startswith("bb.printf_parse")
      or F.getName().startswith("bb.vasnprintf")) {
    return false;
  }

  // This is a hack to prevent clashes between LLVM's `opt` arguments and
  // clangTooling's CommonOptionParser arguments.
  // At this point opt's arguments have already been parsed, so there should
  // be no problem in clearing the map and let clangTooling reinitialize it
  // with its own stuff.
  cl::getRegisteredOptions().clear();

  for (Function &F : *F.getParent()) {
    for (BasicBlock &BB : F) {
      if (!F.getName().startswith("bb."))
        continue;

      std::vector<Instruction *> ToErase;
      for (Instruction &I : BB)
        if (auto *C = dyn_cast<CallInst>(&I))
          if (getCallee(C)->getName() == "newpc")
            ToErase.push_back(C);

      for (Instruction *I : ToErase)
        I->eraseFromParent();
    }
  }

  processFunction(F);

  auto &RestructureCFGAnalysis = getAnalysis<RestructureCFG>();
  ASTTree &CombedCFGAST = RestructureCFGAnalysis.getAST();

  // Here we build the artificial command line for clang tooling
  std::vector<const char *> ArgV = {
    "revng-c",
    "/dev/null", // use /dev/null as input file to start from empty AST
    "--", // separator between tool arguments and clang arguments
    "-xc", // tell clang to compile C language
    "-std=c11", // tell clang to compile C11
  };
  int ArgC = ArgV.size();
  CommonOptionsParser OptionParser(ArgC, ArgV.data(), RevNgCategory);
  ClangTool RevNg = ClangTool(OptionParser.getCompilations(),
                              OptionParser.getSourcePathList());

  DecompilationAction Decompilation(F, CombedCFGAST, std::move(Out));
  using FactoryUniquePtr = std::unique_ptr<FrontendActionFactory>;
  FactoryUniquePtr Factory = newFrontendActionFactory(&Decompilation);
  RevNg.run(Factory.get());

  return true;
}
