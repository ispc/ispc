#include "FastMath.h"
#include "builtins-decl.h"

// Enable all fast math flags. This improves the performance of generated
// codes in some cases but it is much more dangerous. Indeed, it assumes there
// are no NaN, Inf, negative zeros or subnormal values computed by the target 
// code. If this assumption is broken, then LLVM store poison values in the 
// output so the target code basically has an undefined behaviour and may do
// completely crazy things (very hard to debug for ISPC users).
// See https://github.com/iree-org/iree/issues/19743 for more information.
//#define AGGRESSIVE_FAST_MATH_OPT

namespace ispc {

bool FastMathPass::optimizeFpInstructions(llvm::BasicBlock &bb) {
    DEBUG_START_BB("FastMath");

    bool modifiedAny = false;

    llvm::FastMathFlags fmFlags;

#ifdef AGGRESSIVE_FAST_MATH_OPT
    fmFlags.setFast(true);
#else
    // The three first are required to perform basic FP math reassociation.
    fmFlags.setAllowContract(true);
    fmFlags.setAllowReassoc(true);
    fmFlags.setNoSignedZeros(true);
    fmFlags.setAllowReciprocal(true);
#endif

    // Note: we do modify instruction list during the traversal, so the iterator
    // is moved forward before the instruction is processed.
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e;) {
        llvm::BasicBlock::iterator curIter = iter++;
        llvm::Instruction *inst = &*curIter;
        llvm::BinaryOperator *binOpInst = llvm::dyn_cast<llvm::BinaryOperator>(inst);
        llvm::FPMathOperator *fpMathInst = llvm::dyn_cast<llvm::FPMathOperator>(inst);
        if (binOpInst && fpMathInst) {
            binOpInst->setFastMathFlags(fmFlags);
            modifiedAny = true;
        }
    }

    DEBUG_END_BB("FastMath");

    return modifiedAny;
}

llvm::PreservedAnalyses FastMathPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::TimeTraceScope FuncScope("FastMathPass::run", F.getName());
    bool modifiedAny = false;

    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= optimizeFpInstructions(BB);
    }

    if (!modifiedAny) {
        // No changes, all analyses are preserved.
        return llvm::PreservedAnalyses::all();
    }

    llvm::PreservedAnalyses PA;
    PA.preserveSet<llvm::CFGAnalyses>();
    return PA;
}

} // namespace ispc
