/*
  Copyright (c) 2010-2021, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @file ctx.cpp
    @brief Implementation of the FunctionEmitContext class
*/

#include "ctx.h"
#include "expr.h"
#include "func.h"
#include "llvmutil.h"
#include "module.h"
#include "stmt.h"
#include "sym.h"
#include "type.h"
#include "util.h"

#include <map>

#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>

#ifdef ISPC_XE_ENABLED
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#endif

namespace ispc {

/** This is a small utility structure that records information related to one
    level of nested control flow.  It's mostly used in correctly restoring
    the mask and other state as we exit control flow nesting levels.
*/
struct CFInfo {
    /** Returns a new instance of the structure that represents entering an
        'if' statement */
    static CFInfo *GetIf(bool isUniform, bool isUniformEmulated, llvm::Value *savedMask);

    /** Returns a new instance of the structure that represents entering a
        loop. */
    static CFInfo *GetLoop(bool isUniform, bool isUniformEmulated, llvm::BasicBlock *breakTarget,
                           llvm::BasicBlock *continueTarget, llvm::Value *savedBreakLanesPtr,
                           llvm::Value *savedContinueLanesPtr, llvm::Value *savedMask,
                           llvm::Value *savedBlockEntryMask);

    static CFInfo *GetForeach(bool isUniformEmulated, FunctionEmitContext::ForeachType ft,
                              llvm::BasicBlock *breakTarget, llvm::BasicBlock *continueTarget,
                              llvm::Value *savedBreakLanesPtr, llvm::Value *savedContinueLanesPtr,
                              llvm::Value *savedMask, llvm::Value *savedBlockEntryMask);

    static CFInfo *GetSwitch(bool isUniform, bool isUniformEmulated, llvm::BasicBlock *breakTarget,
                             llvm::BasicBlock *continueTarget, llvm::Value *savedBreakLanesPtr,
                             llvm::Value *savedContinueLanesPtr, llvm::Value *savedMask,
                             llvm::Value *savedBlockEntryMask, llvm::Value *switchExpr,
                             llvm::Value *savedFallThroughMaskPtr, llvm::BasicBlock *bbDefault,
                             const std::vector<std::pair<int, llvm::BasicBlock *>> *bbCases,
                             const std::map<llvm::BasicBlock *, llvm::BasicBlock *> *bbNext, bool scUniform);

    bool IsIf() { return type == If; }
    bool IsLoop() { return type == Loop; }
    bool IsForeach() { return (type == ForeachRegular || type == ForeachActive || type == ForeachUnique); }
    bool IsSwitch() { return type == Switch; }
    bool IsVarying() { return !isUniform; }
    bool IsUniform() { return isUniform; }
    bool IsUniformEmulated() { return isUniformEmulated; }

    enum CFType { If, Loop, ForeachRegular, ForeachActive, ForeachUnique, Switch };
    CFType type;
    bool isUniform;
    bool isUniformEmulated;
    llvm::BasicBlock *savedBreakTarget, *savedContinueTarget;
    llvm::Value *savedBreakLanesPtr, *savedContinueLanesPtr;
    llvm::Value *savedMask, *savedBlockEntryMask;
    llvm::Value *savedSwitchExpr;
    llvm::Value *savedSwitchFallThroughMaskPtr;
    llvm::BasicBlock *savedDefaultBlock;
    const std::vector<std::pair<int, llvm::BasicBlock *>> *savedCaseBlocks;
    const std::map<llvm::BasicBlock *, llvm::BasicBlock *> *savedNextBlocks;
    bool savedSwitchConditionWasUniform;

  private:
    CFInfo(CFType t, bool uniformIf, bool uniformEmu, llvm::Value *sm) {
        Assert(t == If);
        type = t;
        isUniform = uniformIf;
        isUniformEmulated = uniformEmu;
        savedBreakTarget = savedContinueTarget = NULL;
        savedBreakLanesPtr = savedContinueLanesPtr = NULL;
        savedMask = savedBlockEntryMask = sm;
        savedSwitchExpr = NULL;
        savedSwitchFallThroughMaskPtr = NULL;
        savedDefaultBlock = NULL;
        savedCaseBlocks = NULL;
        savedNextBlocks = NULL;
        savedSwitchConditionWasUniform = false;
    }
    CFInfo(CFType t, bool iu, bool uniformEmulated, llvm::BasicBlock *bt, llvm::BasicBlock *ct, llvm::Value *sb,
           llvm::Value *sc, llvm::Value *sm, llvm::Value *lm, llvm::Value *sse = NULL, llvm::Value *ssftmp = NULL,
           llvm::BasicBlock *bbd = NULL, const std::vector<std::pair<int, llvm::BasicBlock *>> *bbc = NULL,
           const std::map<llvm::BasicBlock *, llvm::BasicBlock *> *bbn = NULL, bool scu = false) {
        Assert(t == Loop || t == Switch);
        type = t;
        isUniform = iu;
        isUniformEmulated = uniformEmulated;
        savedBreakTarget = bt;
        savedContinueTarget = ct;
        savedBreakLanesPtr = sb;
        savedContinueLanesPtr = sc;
        savedMask = sm;
        savedBlockEntryMask = lm;
        savedSwitchExpr = sse;
        savedSwitchFallThroughMaskPtr = ssftmp;
        savedDefaultBlock = bbd;
        savedCaseBlocks = bbc;
        savedNextBlocks = bbn;
        savedSwitchConditionWasUniform = scu;
    }
    CFInfo(CFType t, bool uniformEmulated, llvm::BasicBlock *bt, llvm::BasicBlock *ct, llvm::Value *sb, llvm::Value *sc,
           llvm::Value *sm, llvm::Value *lm) {
        Assert(t == ForeachRegular || t == ForeachActive || t == ForeachUnique);
        type = t;
        isUniform = uniformEmulated;
        isUniformEmulated = uniformEmulated;
        savedBreakTarget = bt;
        savedContinueTarget = ct;
        savedBreakLanesPtr = sb;
        savedContinueLanesPtr = sc;
        savedMask = sm;
        savedBlockEntryMask = lm;
        savedSwitchExpr = NULL;
        savedSwitchFallThroughMaskPtr = NULL;
        savedDefaultBlock = NULL;
        savedCaseBlocks = NULL;
        savedNextBlocks = NULL;
        savedSwitchConditionWasUniform = false;
    }
};

CFInfo *CFInfo::GetIf(bool isUniform, bool isUniformEmulated, llvm::Value *savedMask) {
    return new CFInfo(If, isUniform, isUniformEmulated, savedMask);
}

CFInfo *CFInfo::GetLoop(bool isUniform, bool isUniformEmulated, llvm::BasicBlock *breakTarget,
                        llvm::BasicBlock *continueTarget, llvm::Value *savedBreakLanesPtr,
                        llvm::Value *savedContinueLanesPtr, llvm::Value *savedMask, llvm::Value *savedBlockEntryMask) {
    return new CFInfo(Loop, isUniform, isUniformEmulated, breakTarget, continueTarget, savedBreakLanesPtr,
                      savedContinueLanesPtr, savedMask, savedBlockEntryMask);
}

CFInfo *CFInfo::GetForeach(bool isUniformEmulated, FunctionEmitContext::ForeachType ft, llvm::BasicBlock *breakTarget,
                           llvm::BasicBlock *continueTarget, llvm::Value *savedBreakLanesPtr,
                           llvm::Value *savedContinueLanesPtr, llvm::Value *savedMask, llvm::Value *savedForeachMask) {
    CFType cfType;
    switch (ft) {
    case FunctionEmitContext::FOREACH_REGULAR:
        cfType = ForeachRegular;
        break;
    case FunctionEmitContext::FOREACH_ACTIVE:
        cfType = ForeachActive;
        break;
    case FunctionEmitContext::FOREACH_UNIQUE:
        cfType = ForeachUnique;
        break;
    default:
        FATAL("Unhandled foreach type");
        return NULL;
    }

    return new CFInfo(cfType, isUniformEmulated, breakTarget, continueTarget, savedBreakLanesPtr, savedContinueLanesPtr,
                      savedMask, savedForeachMask);
}

CFInfo *CFInfo::GetSwitch(bool isUniform, bool isUniformEmulated, llvm::BasicBlock *breakTarget,
                          llvm::BasicBlock *continueTarget, llvm::Value *savedBreakLanesPtr,
                          llvm::Value *savedContinueLanesPtr, llvm::Value *savedMask, llvm::Value *savedBlockEntryMask,
                          llvm::Value *savedSwitchExpr, llvm::Value *savedSwitchFallThroughMaskPtr,
                          llvm::BasicBlock *savedDefaultBlock,
                          const std::vector<std::pair<int, llvm::BasicBlock *>> *savedCases,
                          const std::map<llvm::BasicBlock *, llvm::BasicBlock *> *savedNext,
                          bool savedSwitchConditionUniform) {
    return new CFInfo(Switch, isUniform, isUniformEmulated, breakTarget, continueTarget, savedBreakLanesPtr,
                      savedContinueLanesPtr, savedMask, savedBlockEntryMask, savedSwitchExpr,
                      savedSwitchFallThroughMaskPtr, savedDefaultBlock, savedCases, savedNext,
                      savedSwitchConditionUniform);
}

///////////////////////////////////////////////////////////////////////////

FunctionEmitContext::FunctionEmitContext(Function *func, Symbol *funSym, llvm::Function *lf, SourcePos firstStmtPos) {
    function = func;
    llvmFunction = lf;
    switchConditionWasUniform = false;

    /* Create a new basic block to store all of the allocas */
    allocaBlock = llvm::BasicBlock::Create(*g->ctx, "allocas", llvmFunction, 0);
    bblock = llvm::BasicBlock::Create(*g->ctx, "entry", llvmFunction, 0);
    /* But jump from it immediately into the real entry block */
    llvm::BranchInst::Create(bblock, allocaBlock);

    funcStartPos = funSym->pos;

    internalMaskPointer = AllocaInst(LLVMTypes::MaskType, "internal_mask_memory");
    StoreInst(LLVMMaskAllOn, internalMaskPointer);

    // If the function doesn't have __mask in parameters, there is no need to
    // have function mask
    if (((func->GetType()->isExported || func->GetType()->IsISPCExternal()) &&
         (lf->getFunctionType()->getNumParams() == func->GetType()->GetNumParameters())) ||
        (func->GetType()->isUnmasked) || func->GetType()->isTask) {
        functionMaskValue = NULL;
        fullMaskPointer = NULL;
    } else {
        functionMaskValue = LLVMMaskAllOn;
        fullMaskPointer = AllocaInst(LLVMTypes::MaskType, "full_mask_memory");
        StoreInst(LLVMMaskAllOn, fullMaskPointer);
    }

    blockEntryMask = NULL;
    breakLanesPtr = continueLanesPtr = NULL;
    breakTarget = continueTarget = NULL;

    switchExpr = NULL;
    caseBlocks = NULL;
    defaultBlock = NULL;
    nextBlocks = NULL;

    returnedLanesPtr = AllocaInst(LLVMTypes::MaskType, "returned_lanes_memory");
    StoreInst(LLVMMaskAllOff, returnedLanesPtr);

    launchedTasks = false;
    launchGroupHandlePtr = AllocaInst(LLVMTypes::VoidPointerType, "launch_group_handle");
    StoreInst(llvm::Constant::getNullValue(LLVMTypes::VoidPointerType), launchGroupHandlePtr);

    disableGSWarningCount = 0;

    const Type *returnType = function->GetReturnType();
    if (!returnType || returnType->IsVoidType())
        returnValuePtr = NULL;
    else {
        returnValuePtr = AllocaInst(returnType, "return_value_memory");
    }

#ifdef ISPC_XE_ENABLED
    if (emitXeHardwareMask()) {
        /* Create return point for Xe */
        returnPoint = llvm::BasicBlock::Create(*g->ctx, "return_point", llvmFunction, 0);
        /* Load return value and return it */
        if (returnValuePtr != NULL) {
            // We have value(s) to return; load them from their storage
            // location
            // Note that LoadInst() needs to be used instead of direct llvm instruction generation
            // to handle correctly bool values (they need extra convertion, as memory representation
            // is i8, while in SSa form they are l1)
            auto bb = GetCurrentBasicBlock();
            SetCurrentBasicBlock(returnPoint);
            llvm::Value *retVal = LoadInst(returnValuePtr, returnType, "return_value");
            SetCurrentBasicBlock(bb);
            // llvm::Value *retVal = new llvm::LoadInst(returnValuePtr, "return_value", returnPoint);
            llvm::ReturnInst::Create(*g->ctx, retVal, returnPoint);
        } else {
            llvm::ReturnInst::Create(*g->ctx, returnPoint);
        }
    }
#endif

    if (g->opt.disableMaskAllOnOptimizations) {
        // This is really disgusting.  We want to be able to fool the
        // compiler to not be able to reason that the mask is all on, but
        // we don't want to pay too much of a price at the start of each
        // function to do so.
        //
        // Therefore: first, we declare a module-static __all_on_mask
        // variable that will hold an "all on" mask value.  At the start of
        // each function, we'll load its value and call SetInternalMaskAnd
        // with the result to set the current internal execution mask.
        // (This is a no-op at runtime.)
        //
        // Then, to fool the optimizer that maybe the value of
        // __all_on_mask can't be guaranteed to be "all on", we emit a
        // dummy function that sets __all_on_mask be "all off".  (That
        // function is never actually called.)
        llvm::Value *globalAllOnMaskPtr = m->module->getNamedGlobal("__all_on_mask");
        if (globalAllOnMaskPtr == NULL) {
            globalAllOnMaskPtr =
                new llvm::GlobalVariable(*m->module, LLVMTypes::MaskType, false, llvm::GlobalValue::InternalLinkage,
                                         LLVMMaskAllOn, "__all_on_mask");

            char buf[256];
            snprintf(buf, sizeof(buf), "__off_all_on_mask_%s", g->target->GetISAString());

            llvm::FunctionCallee offFuncCallee = m->module->getOrInsertFunction(buf, LLVMTypes::VoidType);
            llvm::Constant *offFunc = llvm::cast<llvm::Constant>(offFuncCallee.getCallee());
            AssertPos(currentPos, llvm::isa<llvm::Function>(offFunc));
            llvm::BasicBlock *offBB = llvm::BasicBlock::Create(*g->ctx, "entry", (llvm::Function *)offFunc, 0);
            llvm::StoreInst *inst = new llvm::StoreInst(LLVMMaskAllOff, globalAllOnMaskPtr, offBB);
            if (g->opt.forceAlignedMemory) {
                inst->setAlignment(llvm::MaybeAlign(g->target->getNativeVectorAlignment()).valueOrOne());
            }
            llvm::ReturnInst::Create(*g->ctx, offBB);
        }

        llvm::Value *allOnMask = LoadInst(globalAllOnMaskPtr, NULL, "all_on_mask");
        SetInternalMaskAnd(LLVMMaskAllOn, allOnMask);
    }

    if (m->diBuilder) {
        currentPos = funSym->pos;

        /* If debugging is enabled, tell the debug information emission
           code about this new function */
        diFile = funcStartPos.GetDIFile();
        diSpace = funcStartPos.GetDINamespace();
        llvm::DIScope *scope = m->diCompileUnit;
        llvm::DIType *diSubprogramType = NULL;

        const FunctionType *functionType = function->GetType();
        if (functionType == NULL)
            AssertPos(currentPos, m->errorCount > 0);
        else {
            diSubprogramType = functionType->GetDIType(scope);
            /*#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6 // 3.2, 3.3, 3.4, 3.5, 3.6
                        AssertPos(currentPos, diSubprogramType.Verify());
            #else // LLVM 3.7+
                        // comming soon
            #endif*/
        }
        /* LLVM 4.0+ */
        Assert(llvm::isa<llvm::DISubroutineType>(diSubprogramType));
        llvm::DISubroutineType *diSubprogramType_n = llvm::cast<llvm::DISubroutineType>(diSubprogramType);
        llvm::DINode::DIFlags flags = llvm::DINode::FlagPrototyped;

        std::string mangledName = std::string(llvmFunction->getName());
        if (mangledName == funSym->name)
            mangledName = "";

        bool isStatic = (funSym->storageClass == SC_STATIC);
        bool isOptimized = (g->opt.level > 0);
        int firstLine = funcStartPos.first_line;

        /* isDefinition is always set to 'true' */
        llvm::DISubprogram::DISPFlags SPFlags = llvm::DISubprogram::SPFlagDefinition;
        if (isOptimized)
            SPFlags |= llvm::DISubprogram::SPFlagOptimized;
        if (isStatic)
            SPFlags |= llvm::DISubprogram::SPFlagLocalToUnit;

        diSubprogram = m->diBuilder->createFunction(diSpace /* scope */, funSym->name, mangledName, diFile, firstLine,
                                                    diSubprogramType_n, firstLine, flags, SPFlags);
        llvmFunction->setSubprogram(diSubprogram);

        /* And start a scope representing the initial function scope */
        StartScope();
    } else {
        diSubprogram = NULL;
        diFile = NULL;
        diSpace = NULL;
    }
}

FunctionEmitContext::~FunctionEmitContext() {
    AssertPos(currentPos, controlFlowInfo.size() == 0);
    AssertPos(currentPos, debugScopes.size() == (m->diBuilder ? 1 : 0));
}

const Function *FunctionEmitContext::GetFunction() const { return function; }

llvm::BasicBlock *FunctionEmitContext::GetCurrentBasicBlock() { return bblock; }

void FunctionEmitContext::SetCurrentBasicBlock(llvm::BasicBlock *bb) { bblock = bb; }

llvm::Value *FunctionEmitContext::GetFunctionMask() { return fullMaskPointer ? functionMaskValue : LLVMMaskAllOn; }

llvm::Value *FunctionEmitContext::GetInternalMask() { return LoadInst(internalMaskPointer, NULL, "load_mask"); }

llvm::Value *FunctionEmitContext::GetFullMask() {
    return fullMaskPointer ? BinaryOperator(llvm::Instruction::And, GetInternalMask(), functionMaskValue,
                                            "internal_mask&function_mask")
                           : GetInternalMask();
}

llvm::Value *FunctionEmitContext::GetFullMaskPointer() {
    return fullMaskPointer ? fullMaskPointer : internalMaskPointer;
}

void FunctionEmitContext::SetFunctionMask(llvm::Value *value) {
    if (fullMaskPointer != NULL) {
        functionMaskValue = value;
        if (bblock != NULL)
            StoreInst(GetFullMask(), fullMaskPointer);
    }
}

void FunctionEmitContext::SetBlockEntryMask(llvm::Value *value) { blockEntryMask = value; }

void FunctionEmitContext::SetInternalMask(llvm::Value *value) {
    StoreInst(value, internalMaskPointer);
    // kludge so that __mask returns the right value in ispc code.
    if (fullMaskPointer)
        StoreInst(GetFullMask(), fullMaskPointer);
}

void FunctionEmitContext::SetInternalMaskAnd(llvm::Value *oldMask, llvm::Value *test) {
    llvm::Value *mask = BinaryOperator(llvm::Instruction::And, oldMask, test, "oldMask&test");
    SetInternalMask(mask);
}

void FunctionEmitContext::SetInternalMaskAndNot(llvm::Value *oldMask, llvm::Value *test) {
    llvm::Value *notTest = BinaryOperator(llvm::Instruction::Xor, test, LLVMMaskAllOn, "~test");
    llvm::Value *mask = BinaryOperator(llvm::Instruction::And, oldMask, notTest, "oldMask&~test");
    SetInternalMask(mask);
}

llvm::Instruction *FunctionEmitContext::BranchIfMaskAny(llvm::BasicBlock *btrue, llvm::BasicBlock *bfalse) {
    AssertPos(currentPos, bblock != NULL);
    llvm::Value *any = Any(GetFullMask());
    llvm::Instruction *bInst = BranchInst(btrue, bfalse, any);
    // It's illegal to add any additional instructions to the basic block
    // now that it's terminated, so set bblock to NULL to be safe
    bblock = NULL;
    return bInst;
}

void FunctionEmitContext::BranchIfMaskAll(llvm::BasicBlock *btrue, llvm::BasicBlock *bfalse) {
    AssertPos(currentPos, bblock != NULL);
    llvm::Value *all = All(GetFullMask());
    BranchInst(btrue, bfalse, all);
    // It's illegal to add any additional instructions to the basic block
    // now that it's terminated, so set bblock to NULL to be safe
    bblock = NULL;
}

void FunctionEmitContext::BranchIfMaskNone(llvm::BasicBlock *btrue, llvm::BasicBlock *bfalse) {
    AssertPos(currentPos, bblock != NULL);
    // switch sense of true/false bblocks
    BranchIfMaskAny(bfalse, btrue);
    // It's illegal to add any additional instructions to the basic block
    // now that it's terminated, so set bblock to NULL to be safe
    bblock = NULL;
}

void FunctionEmitContext::StartUniformIf(bool emulateUniform) {
    controlFlowInfo.push_back(CFInfo::GetIf(true, emulateUniform, GetInternalMask()));
}

void FunctionEmitContext::StartVaryingIf(llvm::Value *oldMask) {
    controlFlowInfo.push_back(CFInfo::GetIf(false, false, oldMask));
}

void FunctionEmitContext::EndIf() {
    CFInfo *ci = popCFState();
    // Make sure we match up with a Start{Uniform,Varying}If().
    AssertPos(currentPos, ci->IsIf());

    // 'uniform' ifs don't change the mask so we only need to restore the
    // mask going into the if for 'varying' if statements
    if (ci->IsUniform() || bblock == NULL)
        return;

    // We can't just restore the mask as it was going into the 'if'
    // statement.  First we have to take into account any program
    // instances that have executed 'return' statements; the restored
    // mask must be off for those lanes.
    restoreMaskGivenReturns(ci->savedMask);

    // If the 'if' statement is inside a loop with a 'varying'
    // condition, we also need to account for any break or continue
    // statements that executed inside the 'if' statmeent; we also must
    // leave the lane masks for the program instances that ran those
    // off after we restore the mask after the 'if'.  The code below
    // ends up being optimized out in the case that there were no break
    // or continue statements (and breakLanesPtr and continueLanesPtr
    // have their initial 'all off' values), so we don't need to check
    // for that here.
    //
    // There are three general cases to deal with here:
    // - Loops: both break and continue are allowed, and thus the corresponding
    //   lane mask pointers are non-NULL
    // - Foreach: only continueLanesPtr may be non-NULL
    // - Switch: only breakLanesPtr may be non-NULL
    if (continueLanesPtr != NULL || breakLanesPtr != NULL) {
        // We want to compute:
        // newMask = (oldMask & ~(breakLanes | continueLanes)),
        // treading breakLanes or continueLanes as "all off" if the
        // corresponding pointer is NULL.
        llvm::Value *bcLanes = NULL;

        if (continueLanesPtr != NULL)
            bcLanes = LoadInst(continueLanesPtr, NULL, "continue_lanes");
        else
            bcLanes = LLVMMaskAllOff;

        if (breakLanesPtr != NULL) {
            llvm::Value *breakLanes = LoadInst(breakLanesPtr, NULL, "break_lanes");
            bcLanes = BinaryOperator(llvm::Instruction::Or, bcLanes, breakLanes, "|break_lanes");
        }

        llvm::Value *notBreakOrContinue =
            BinaryOperator(llvm::Instruction::Xor, bcLanes, LLVMMaskAllOn, "!(break|continue)_lanes");
        llvm::Value *oldMask = GetInternalMask();
        llvm::Value *newMask = BinaryOperator(llvm::Instruction::And, oldMask, notBreakOrContinue, "new_mask");
        SetInternalMask(newMask);
    }
}

void FunctionEmitContext::StartLoop(llvm::BasicBlock *bt, llvm::BasicBlock *ct, bool uniformCF,
                                    bool isEmulatedUniform) {
    // Store the current values of various loop-related state so that we
    // can restore it when we exit this loop.
    llvm::Value *oldMask = GetInternalMask();
    controlFlowInfo.push_back(CFInfo::GetLoop(uniformCF, isEmulatedUniform, breakTarget, continueTarget, breakLanesPtr,
                                              continueLanesPtr, oldMask, blockEntryMask));
    if (uniformCF)
        // If the loop has a uniform condition, we don't need to track
        // which lanes 'break' or 'continue'; all of the running ones go
        // together, so we just jump
        breakLanesPtr = continueLanesPtr = NULL;
    else {
        // For loops with varying conditions, allocate space to store masks
        // that record which lanes have done these
        continueLanesPtr = AllocaInst(LLVMTypes::MaskType, "continue_lanes_memory");
        StoreInst(LLVMMaskAllOff, continueLanesPtr);
        breakLanesPtr = AllocaInst(LLVMTypes::MaskType, "break_lanes_memory");
        StoreInst(LLVMMaskAllOff, breakLanesPtr);
    }

    breakTarget = bt;
    continueTarget = ct;
    blockEntryMask = NULL; // this better be set by the loop!
}

void FunctionEmitContext::EndLoop() {
    CFInfo *ci = popCFState();
    AssertPos(currentPos, ci->IsLoop());

    if (!ci->IsUniform())
        // If the loop had a 'uniform' test, then it didn't make any
        // changes to the mask so there's nothing to restore.  If it had a
        // varying test, we need to restore the mask to what it was going
        // into the loop, but still leaving off any lanes that executed a
        // 'return' statement.
        restoreMaskGivenReturns(ci->savedMask);
}

void FunctionEmitContext::StartForeach(ForeachType ft, bool isEmulatedUniform) {
    // Issue an error if we're in a nested foreach...
    if (ft == FOREACH_REGULAR) {
        for (int i = 0; i < (int)controlFlowInfo.size(); ++i) {
            if (controlFlowInfo[i]->type == CFInfo::ForeachRegular) {
                Error(currentPos, "Nested \"foreach\" statements are currently "
                                  "illegal.");
                break;
                // Don't return here, however, and in turn allow the caller to
                // do the rest of its codegen and then call EndForeach()
                // normally--the idea being that this gives a chance to find
                // any other errors inside the body of the foreach loop...
            }
        }
    }

    // Store the current values of various loop-related state so that we
    // can restore it when we exit this loop.
    llvm::Value *oldMask = GetInternalMask();
    controlFlowInfo.push_back(CFInfo::GetForeach(isEmulatedUniform, ft, breakTarget, continueTarget, breakLanesPtr,
                                                 continueLanesPtr, oldMask, blockEntryMask));
    breakLanesPtr = NULL;
    breakTarget = NULL;

    continueLanesPtr = NULL;
    if (!isEmulatedUniform) {
        continueLanesPtr = AllocaInst(LLVMTypes::MaskType, "foreach_continue_lanes");
        StoreInst(LLVMMaskAllOff, continueLanesPtr);
    }

    continueTarget = NULL; // should be set by SetContinueTarget()

    blockEntryMask = NULL;
}

void FunctionEmitContext::EndForeach() {
    CFInfo *ci = popCFState();
    AssertPos(currentPos, ci->IsForeach());
}

void FunctionEmitContext::restoreMaskGivenReturns(llvm::Value *oldMask) {
    if (!bblock)
        return;

    // Restore the mask to the given old mask, but leave off any lanes that
    // executed a return statement.
    // newMask = (oldMask & ~returnedLanes)
    llvm::Value *returnedLanes = LoadInst(returnedLanesPtr, NULL, "returned_lanes");
    llvm::Value *notReturned = BinaryOperator(llvm::Instruction::Xor, returnedLanes, LLVMMaskAllOn, "~returned_lanes");
    llvm::Value *newMask = BinaryOperator(llvm::Instruction::And, oldMask, notReturned, "new_mask");
    SetInternalMask(newMask);
}

/** Returns "true" if the first enclosing non-if control flow expression is
    a "switch" statement.
*/
bool FunctionEmitContext::inSwitchStatement() const {
    // Go backwards through controlFlowInfo, since we add new nested scopes
    // to the back.
    int i = controlFlowInfo.size() - 1;
    while (i >= 0 && controlFlowInfo[i]->IsIf())
        --i;
    // Got to the first non-if (or end of CF info)
    if (i == -1)
        return false;
    return controlFlowInfo[i]->IsSwitch();
}

void FunctionEmitContext::Break(bool doCoherenceCheck) {
    if (breakTarget == NULL) {
        Error(currentPos, "\"break\" statement is illegal outside of "
                          "for/while/do loops and \"switch\" statements.");
        return;
    }
    AssertPos(currentPos, controlFlowInfo.size() > 0);

    if (bblock == NULL)
        return;

    if (inSwitchStatement() == true && switchConditionWasUniform == true && ifsInCFAllUniform(CFInfo::Switch)) {
        // We know that all program instances are executing the break, so
        // just jump to the block immediately after the switch.
        AssertPos(currentPos, breakTarget != NULL);
        BranchInst(breakTarget);
        bblock = NULL;
        return;
    }

    // If all of the enclosing 'if' tests in the loop have uniform control
    // flow or if we can tell that the mask is all on, then we can just
    // jump to the break location.
    if (inSwitchStatement() == false && ifsInCFAllUniform(CFInfo::Loop)) {
        BranchInst(breakTarget);
        // Set bblock to NULL since the jump has terminated the basic block
        bblock = NULL;
    } else {
        // Varying switch, uniform switch where the 'break' is under
        // varying control flow, or a loop with varying 'if's above the
        // break.  In these cases, we need to update the mask of the lanes
        // that have executed a 'break' statement:
        // breakLanes = breakLanes | mask
        AssertPos(currentPos, breakLanesPtr != NULL);

        llvm::Value *mask = GetInternalMask();
        llvm::Value *breakMask = LoadInst(breakLanesPtr, NULL, "break_mask");
        llvm::Value *newMask = BinaryOperator(llvm::Instruction::Or, mask, breakMask, "mask|break_mask");
        StoreInst(newMask, breakLanesPtr);

        // Set the current mask to be all off, just in case there are any
        // statements in the same scope after the 'break'.  Most of time
        // this will be optimized away since we'll likely end the scope of
        // an 'if' statement and restore the mask then.
        SetInternalMask(LLVMMaskAllOff);

        if (doCoherenceCheck) {
            if (continueTarget != NULL)
                // If the user has indicated that this is a 'coherent'
                // break statement, then check to see if the mask is all
                // off.  If so, we have to conservatively jump to the
                // continueTarget, not the breakTarget, since part of the
                // reason the mask is all off may be due to 'continue'
                // statements that executed in the current loop iteration.
                jumpIfAllLoopLanesAreDone(continueTarget);
            else if (breakTarget != NULL)
                // Similarly handle these for switch statements, where we
                // only have a break target.
                jumpIfAllLoopLanesAreDone(breakTarget);
        }
    }
}

static bool lEnclosingLoopIsForeachActive(const std::vector<CFInfo *> &controlFlowInfo) {
    for (int i = (int)controlFlowInfo.size() - 1; i >= 0; --i) {
        if (controlFlowInfo[i]->type == CFInfo::ForeachActive)
            return true;
    }
    return false;
}

void FunctionEmitContext::Continue(bool doCoherenceCheck) {
    if (!continueTarget) {
        Error(currentPos, "\"continue\" statement illegal outside of "
                          "for/while/do/foreach loops.");
        return;
    }
    AssertPos(currentPos, controlFlowInfo.size() > 0);

    if (ifsInCFAllUniform(CFInfo::Loop) || lEnclosingLoopIsForeachActive(controlFlowInfo)) {
        // Similarly to 'break' statements, we can immediately jump to the
        // continue target if we're only in 'uniform' control flow within
        // loop or if we can tell that the mask is all on.  Here, we can
        // also jump if the enclosing loop is a 'foreach_active' loop, in
        // which case we know that only a single program instance is
        // executing.
        AddInstrumentationPoint("continue: uniform CF, jumped");
        BranchInst(continueTarget);
        bblock = NULL;
    } else {
        // Otherwise update the stored value of which lanes have 'continue'd.
        // continueLanes = continueLanes | mask
        AssertPos(currentPos, continueLanesPtr);
        llvm::Value *mask = GetInternalMask();
        llvm::Value *continueMask = LoadInst(continueLanesPtr, NULL, "continue_mask");
        llvm::Value *newMask = BinaryOperator(llvm::Instruction::Or, mask, continueMask, "mask|continueMask");
        StoreInst(newMask, continueLanesPtr);

        // And set the current mask to be all off in case there are any
        // statements in the same scope after the 'continue'
        SetInternalMask(LLVMMaskAllOff);

        if (doCoherenceCheck)
            // If this is a 'coherent continue' statement, then emit the
            // code to see if all of the lanes are now off due to
            // breaks/continues and jump to the continue target if so.
            jumpIfAllLoopLanesAreDone(continueTarget);
    }
}

/** This function checks to see if all of the 'if' statements (if any)
    between the current scope and the first enclosing loop/switch of given
    control flow type have 'uniform' tests.
 */
bool FunctionEmitContext::ifsInCFAllUniform(int type) const {
    AssertPos(currentPos, controlFlowInfo.size() > 0);
    // Go backwards through controlFlowInfo, since we add new nested scopes
    // to the back.  Stop once we come to the first enclosing control flow
    // structure of the desired type.
    int i = controlFlowInfo.size() - 1;
    while (i >= 0 && controlFlowInfo[i]->type != type) {
        if (controlFlowInfo[i]->isUniform == false)
            // Found a scope due to an 'if' statement with a varying test
            return false;
        --i;
    }
    return true;
}

void FunctionEmitContext::jumpIfAllLoopLanesAreDone(llvm::BasicBlock *target) {
    llvm::Value *allDone = NULL;

    if (breakLanesPtr == NULL) {
        llvm::Value *continued = LoadInst(continueLanesPtr, NULL, "continue_lanes");
        continued = BinaryOperator(llvm::Instruction::And, continued, GetFunctionMask(), "continued&func");
        allDone = MasksAllEqual(continued, blockEntryMask);
    } else {
        // Check to see if (returned lanes | continued lanes | break lanes) is
        // equal to the value of mask at the start of the loop iteration.  If
        // so, everyone is done and we can jump to the given target
        llvm::Value *returned = LoadInst(returnedLanesPtr, NULL, "returned_lanes");
        llvm::Value *breaked = LoadInst(breakLanesPtr, NULL, "break_lanes");
        llvm::Value *finishedLanes = BinaryOperator(llvm::Instruction::Or, returned, breaked, "returned|breaked");
        if (continueLanesPtr != NULL) {
            // It's NULL for "switch" statements...
            llvm::Value *continued = LoadInst(continueLanesPtr, NULL, "continue_lanes");
            finishedLanes =
                BinaryOperator(llvm::Instruction::Or, finishedLanes, continued, "returned|breaked|continued");
        }

        finishedLanes = BinaryOperator(llvm::Instruction::And, finishedLanes, GetFunctionMask(), "finished&func");

        // Do we match the mask at loop or switch statement entry?
        allDone = MasksAllEqual(finishedLanes, blockEntryMask);
    }

    llvm::BasicBlock *bAll = CreateBasicBlock("all_continued_or_breaked");
    llvm::BasicBlock *bNotAll = CreateBasicBlock("not_all_continued_or_breaked");

    BranchInst(bAll, bNotAll, allDone);

    // If so, have an extra basic block along the way to add
    // instrumentation, if the user asked for it.
    bblock = bAll;
    AddInstrumentationPoint("break/continue: all dynamically went");
    BranchInst(target);

    // And set the current basic block to a new one for future instructions
    // for the path where we weren't able to jump
    bblock = bNotAll;
    AddInstrumentationPoint("break/continue: not all went");
}

void FunctionEmitContext::RestoreContinuedLanes() {
    if (continueLanesPtr == NULL)
        return;

    // mask = mask & continueFlags
    llvm::Value *mask = GetInternalMask();
    llvm::Value *continueMask = LoadInst(continueLanesPtr, NULL, "continue_mask");
    llvm::Value *orMask = BinaryOperator(llvm::Instruction::Or, mask, continueMask, "mask|continue_mask");
    SetInternalMask(orMask);

    // continueLanes = 0
    StoreInst(LLVMMaskAllOff, continueLanesPtr);
}

void FunctionEmitContext::ClearBreakLanes() {
    if (breakLanesPtr == NULL)
        return;

    // breakLanes = 0
    StoreInst(LLVMMaskAllOff, breakLanesPtr);
}

void FunctionEmitContext::StartSwitch(bool cfIsUniform, llvm::BasicBlock *bbBreak, bool isEmulatedUniform) {
    llvm::Value *oldMask = GetInternalMask();
    controlFlowInfo.push_back(CFInfo::GetSwitch(cfIsUniform, isEmulatedUniform, breakTarget, continueTarget,
                                                breakLanesPtr, continueLanesPtr, oldMask, blockEntryMask, switchExpr,
                                                switchFallThroughMaskPtr, defaultBlock, caseBlocks, nextBlocks,
                                                switchConditionWasUniform));

    breakLanesPtr = AllocaInst(LLVMTypes::MaskType, "break_lanes_memory");
    StoreInst(LLVMMaskAllOff, breakLanesPtr);
    breakTarget = bbBreak;

    continueLanesPtr = NULL;
    continueTarget = NULL;
    blockEntryMask = NULL;

    // These will be set by the SwitchInst() method
    switchExpr = NULL;
    switchFallThroughMaskPtr = NULL;
    defaultBlock = NULL;
    caseBlocks = NULL;
    nextBlocks = NULL;
}

void FunctionEmitContext::EndSwitch() {
    AssertPos(currentPos, bblock != NULL);

    CFInfo *ci = popCFState();
    if (ci->IsVarying() && bblock != NULL)
        restoreMaskGivenReturns(ci->savedMask);
}

/** Emit code to check for an "all off" mask before the code for a
    case or default label in a "switch" statement.
 */
void FunctionEmitContext::addSwitchMaskCheck(llvm::Value *mask) {
    llvm::Value *allOff = None(mask);
    llvm::BasicBlock *bbSome = CreateBasicBlock("case_default_on");

    // Find the basic block for the case or default label immediately after
    // the current one in the switch statement--that's where we want to
    // jump if the mask is all off at this label.
    AssertPos(currentPos, nextBlocks->find(bblock) != nextBlocks->end());
    llvm::BasicBlock *bbNext = nextBlocks->find(bblock)->second;

    // Jump to the next one of the mask is all off; otherwise jump to the
    // newly created block that will hold the actual code for this label.
    BranchInst(bbNext, bbSome, allOff);
    SetCurrentBasicBlock(bbSome);
}

/** Returns the execution mask at entry to the first enclosing "switch"
    statement. */
llvm::Value *FunctionEmitContext::getMaskAtSwitchEntry() {
    AssertPos(currentPos, controlFlowInfo.size() > 0);
    int i = controlFlowInfo.size() - 1;
    while (i >= 0 && controlFlowInfo[i]->type != CFInfo::Switch)
        --i;
    AssertPos(currentPos, i != -1);
    return controlFlowInfo[i]->savedMask;
}

void FunctionEmitContext::EmitDefaultLabel(bool checkMask, SourcePos pos) {
    if (inSwitchStatement() == false) {
        Error(pos, "\"default\" label illegal outside of \"switch\" "
                   "statement.");
        return;
    }

    // If there's a default label in the switch, a basic block for it
    // should have been provided in the previous call to SwitchInst().
    AssertPos(currentPos, defaultBlock != NULL);

#ifdef ISPC_XE_ENABLED
    llvm::BasicBlock *bbDefaultImpl = NULL;
    if (emitXeHardwareMask()) {
        // Create basic block with actual default implementation
        bbDefaultImpl = CreateBasicBlock("default_impl", defaultBlock);
    }
#endif

    if (bblock != NULL) {
        // The previous case in the switch fell through, or we're in a
        // varying switch; terminate the current block with a jump to the
        // block for the code for the default label.
#ifdef ISPC_XE_ENABLED
        if (emitXeHardwareMask() && !inXeSimdCF()) {
            // Skip check, branch directly to implementation
            BranchInst(bbDefaultImpl);
        } else
#endif
            BranchInst(defaultBlock);
    }
    SetCurrentBasicBlock(defaultBlock);

#ifdef ISPC_XE_ENABLED
    if (switchConditionWasUniform && emitXeHardwareMask()) {
        // Find next basic block after default
        auto iter = nextBlocks->find(defaultBlock);
        AssertPos(currentPos, iter != nextBlocks->end());
        llvm::BasicBlock *bbNext = iter->second;

        llvm::Value *testVal = llvm::isa<llvm::VectorType>(switchExpr->getType()) ? LLVMMaskAllOn : LLVMTrue;

        // We check only cases after default:
        // EM is turned off for previous ones (or not in case off fall through)
        // Find case value for the next case
        auto caseBlocksIt = caseBlocks->begin();
        for (auto e = caseBlocks->end(); (caseBlocksIt != e) && (caseBlocksIt->second != bbNext); ++caseBlocksIt)
            ;

        for (auto e = caseBlocks->end(); caseBlocksIt != e; ++caseBlocksIt) {
            int value = caseBlocksIt->first;
            llvm::Value *val = NULL;
            if (llvm::isa<llvm::VectorType>(switchExpr->getType())) {
                val = (switchExpr->getType() == LLVMTypes::Int32VectorType) ? LLVMInt32Vector(value)
                                                                            : LLVMInt64Vector(value);
            } else {
                val = (switchExpr->getType() == LLVMTypes::Int32Type) ? LLVMInt32(value) : LLVMInt64(value);
            }

            // The way to get cmp is the same as under TODO comment below.
            // However, seems like such constructions are transformed to cmp.ne
            // in vISA anyway
            llvm::Value *matchesCaseValue =
                CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, switchExpr, val, "cmp_case_value");
            llvm::Value *notMatchesCaseValue = NotOperator(matchesCaseValue);
            testVal = BinaryOperator(llvm::Instruction::And, testVal, notMatchesCaseValue, "default&~case_match");
        }

        // Don't need to check fall through mask: all lanes that
        // executed on fall through won't fail case checks from
        // above

        // Branch to default/next block. It will set Xe EM
        // for this block and restore mask for turned off lanes after
        // reaching next block
        BranchInst(bbDefaultImpl, bbNext, testVal);
        SetCurrentBasicBlock(bbDefaultImpl);
    }
#endif

    if (switchConditionWasUniform)
        // Nothing more to do for this case; return back to the caller,
        // which will then emit the code for the default case.
        return;

    // For a varying switch, we need to update the execution mask.
    //
    // First, compute the mask that corresponds to which program instances
    // should execute the "default" code; this corresponds to the set of
    // program instances that don't match any of the case statements.
    // Therefore, we generate code that compares the value of the switch
    // expression to the value associated with each of the "case"
    // statements such that the surviving lanes didn't match any of them.
    llvm::Value *matchesDefault = getMaskAtSwitchEntry();
    for (int i = 0; i < (int)caseBlocks->size(); ++i) {
        int value = (*caseBlocks)[i].first;
        llvm::Value *valueVec =
            (switchExpr->getType() == LLVMTypes::Int32VectorType) ? LLVMInt32Vector(value) : LLVMInt64Vector(value);
        // TODO: for AVX2 at least, the following generates better code
        // than doing ICMP_NE and skipping the NotOperator() below; file a
        // LLVM bug?
        llvm::Value *matchesCaseValue =
            CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, switchExpr, valueVec, "cmp_case_value");
        matchesCaseValue = I1VecToBoolVec(matchesCaseValue);

        llvm::Value *notMatchesCaseValue = NotOperator(matchesCaseValue);
        matchesDefault =
            BinaryOperator(llvm::Instruction::And, matchesDefault, notMatchesCaseValue, "default&~case_match");
    }

    // The mask may have some lanes on, which corresponds to the previous
    // label falling through; compute the updated mask by ANDing with the
    // current mask.
    llvm::Value *oldMask = GetInternalMask();
    llvm::Value *newMask = BinaryOperator(llvm::Instruction::Or, oldMask, matchesDefault, "old_mask|matches_default");
    SetInternalMask(newMask);

    if (checkMask)
        addSwitchMaskCheck(newMask);
}

void FunctionEmitContext::EmitCaseLabel(int value, bool checkMask, SourcePos pos) {
    if (inSwitchStatement() == false) {
        Error(pos, "\"case\" label illegal outside of \"switch\" statement.");
        return;
    }

    // Find the basic block for this case statement.
    llvm::BasicBlock *bbCase = NULL;
    AssertPos(currentPos, caseBlocks != NULL);
    for (int i = 0; i < (int)caseBlocks->size(); ++i)
        if ((*caseBlocks)[i].first == value) {
            bbCase = (*caseBlocks)[i].second;
            break;
        }
    AssertPos(currentPos, bbCase != NULL);

#ifdef ISPC_XE_ENABLED
    llvm::BasicBlock *bbCaseImpl = NULL;
    if (emitXeHardwareMask()) {
        // Create basic block with actual case implementation
        bbCaseImpl = CreateBasicBlock(llvm::Twine(bbCase->getName()) + "_impl", bbCase);
    }
#endif

    if (bblock != NULL) {
        // fall through from the previous case
#ifdef ISPC_XE_ENABLED
        if (emitXeHardwareMask() && llvm::isa<llvm::VectorType>(switchExpr->getType())) {
            // EM will be restored after this branch.
            // We need to skip case check for lanes that are
            // turned on at this point.
            StoreInst(XeSimdCFPredicate(LLVMMaskAllOn), switchFallThroughMaskPtr);
        }

        if (emitXeHardwareMask() && !inXeSimdCF()) {
            // Skip check, branch directly to implementation
            BranchInst(bbCaseImpl);
        } else
#endif
            BranchInst(bbCase);
    }
    SetCurrentBasicBlock(bbCase);

#ifdef ISPC_XE_ENABLED
    if (switchConditionWasUniform && emitXeHardwareMask()) {
        // Find the next basic block after this case
        std::map<llvm::BasicBlock *, llvm::BasicBlock *>::const_iterator iter;
        iter = nextBlocks->find(bbCase);
        AssertPos(currentPos, iter != nextBlocks->end());
        llvm::BasicBlock *bbNext = iter->second;

        // Create compare value
        llvm::Value *caseTest = NULL;
        if (llvm::isa<llvm::VectorType>(switchExpr->getType())) {
            // Take fall through lanes to turn them on in the next block
            llvm::Value *fallThroughMask = LoadInst(switchFallThroughMaskPtr, NULL, "fall_through_mask");
            llvm::Value *val =
                (switchExpr->getType() == LLVMTypes::Int32VectorType) ? LLVMInt32Vector(value) : LLVMInt64Vector(value);
            llvm::Value *cmpVal =
                CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, switchExpr, val, "cmp_case_value");
            caseTest = BinaryOperator(llvm::Instruction::Or, cmpVal, fallThroughMask, "case_test");
        } else {
            llvm::Value *val = (switchExpr->getType() == LLVMTypes::Int32Type) ? LLVMInt32(value) : LLVMInt64(value);
            caseTest = CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, switchExpr, val, "case_test");
        }

        // Branch to current case/next block. It will set Xe EM
        // for this block and restore mask for turned off lanes after
        // reaching next block
        BranchInst(bbCaseImpl, bbNext, caseTest);
        SetCurrentBasicBlock(bbCaseImpl);
    }
#endif

    if (switchConditionWasUniform)
        return;

    // update the mask: first, get a mask that indicates which program
    // instances have a value for the switch expression that matches this
    // case statement.
    llvm::Value *valueVec =
        (switchExpr->getType() == LLVMTypes::Int32VectorType) ? LLVMInt32Vector(value) : LLVMInt64Vector(value);
    llvm::Value *matchesCaseValue =
        CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, switchExpr, valueVec, "cmp_case_value");
    matchesCaseValue = I1VecToBoolVec(matchesCaseValue);

    // If a lane was off going into the switch, we don't care if has a
    // value in the switch expression that happens to match this case.
    llvm::Value *entryMask = getMaskAtSwitchEntry();
    matchesCaseValue = BinaryOperator(llvm::Instruction::And, entryMask, matchesCaseValue, "entry_mask&case_match");

    // Take the surviving lanes and turn on the mask for them.
    llvm::Value *oldMask = GetInternalMask();
    llvm::Value *newMask = BinaryOperator(llvm::Instruction::Or, oldMask, matchesCaseValue, "mask|case_match");
    SetInternalMask(newMask);

    if (checkMask)
        addSwitchMaskCheck(newMask);
}

void FunctionEmitContext::SwitchInst(llvm::Value *expr, llvm::BasicBlock *bbDefault,
                                     const std::vector<std::pair<int, llvm::BasicBlock *>> &bbCases,
                                     const std::map<llvm::BasicBlock *, llvm::BasicBlock *> &bbNext) {
    // The calling code should have called StartSwitch() before calling
    // SwitchInst().
    AssertPos(currentPos, controlFlowInfo.size() && controlFlowInfo.back()->IsSwitch());

    switchExpr = expr;
    defaultBlock = bbDefault;
    caseBlocks = new std::vector<std::pair<int, llvm::BasicBlock *>>(bbCases);
    nextBlocks = new std::map<llvm::BasicBlock *, llvm::BasicBlock *>(bbNext);
    switchConditionWasUniform =
        (llvm::isa<llvm::VectorType>(expr->getType()) == false) || (controlFlowInfo.back()->IsUniformEmulated());

    // Do not make LLVM switch for Xe
    if (switchConditionWasUniform == true && !emitXeHardwareMask()) {
        // For a uniform switch condition, just wire things up to the LLVM
        // switch instruction.
        llvm::SwitchInst *s = llvm::SwitchInst::Create(expr, bbDefault, bbCases.size(), bblock);
        for (int i = 0; i < (int)bbCases.size(); ++i) {
            if (expr->getType() == LLVMTypes::Int32Type)
                s->addCase(LLVMInt32(bbCases[i].first), bbCases[i].second);
            else {
                AssertPos(currentPos, expr->getType() == LLVMTypes::Int64Type);
                s->addCase(LLVMInt64(bbCases[i].first), bbCases[i].second);
            }
        }

        AddDebugPos(s);
        // switch is a terminator
        bblock = NULL;
    } else {
        if (emitXeHardwareMask()) {
            // Init fall through mask
            switchFallThroughMaskPtr = AllocaInst(LLVMTypes::MaskType, "fall_through_mask");
            StoreInst(LLVMMaskAllOff, switchFallThroughMaskPtr);
        } else {
            // For a varying switch, we first turn off all lanes of the mask
            SetInternalMask(LLVMMaskAllOff);
        }

        if (nextBlocks->size() > 0) {
            // If there are any labels inside the switch, jump to the first
            // one; any code before the first label won't be executed by
            // anyone.
            std::map<llvm::BasicBlock *, llvm::BasicBlock *>::const_iterator iter;
            iter = nextBlocks->find(NULL);
            AssertPos(currentPos, iter != nextBlocks->end());
            llvm::BasicBlock *bbFirst = iter->second;
            BranchInst(bbFirst);
            bblock = NULL;
        }
    }
}

int FunctionEmitContext::VaryingCFDepth() const {
    int sum = 0;
    for (unsigned int i = 0; i < controlFlowInfo.size(); ++i)
        if (controlFlowInfo[i]->IsVarying())
            ++sum;
    return sum;
}

bool FunctionEmitContext::InForeachLoop() const {
    for (unsigned int i = 0; i < controlFlowInfo.size(); ++i)
        if (controlFlowInfo[i]->IsForeach())
            return true;
    return false;
}

void FunctionEmitContext::DisableGatherScatterWarnings() { ++disableGSWarningCount; }

void FunctionEmitContext::EnableGatherScatterWarnings() { --disableGSWarningCount; }

bool FunctionEmitContext::initLabelBBlocks(ASTNode *node, void *data) {
    LabeledStmt *ls = llvm::dyn_cast<LabeledStmt>(node);
    if (ls == NULL)
        return true;

    FunctionEmitContext *ctx = (FunctionEmitContext *)data;

    if (ctx->labelMap.find(ls->name) != ctx->labelMap.end())
        Error(ls->pos, "Multiple labels named \"%s\" in function.", ls->name.c_str());
    else {
        llvm::BasicBlock *bb = ctx->CreateBasicBlock(ls->name);
        ctx->labelMap[ls->name] = bb;
    }
    return true;
}

void FunctionEmitContext::InitializeLabelMap(Stmt *code) {
    labelMap.erase(labelMap.begin(), labelMap.end());
    WalkAST(code, initLabelBBlocks, NULL, this);
}

llvm::BasicBlock *FunctionEmitContext::GetLabeledBasicBlock(const std::string &label) {
    if (labelMap.find(label) != labelMap.end())
        return labelMap[label];
    else
        return NULL;
}

std::vector<std::string> FunctionEmitContext::GetLabels() {
    // Initialize vector to the right size
    std::vector<std::string> labels(labelMap.size());

    // Iterate through labelMap and grab only the keys
    std::map<std::string, llvm::BasicBlock *>::iterator iter;
    for (iter = labelMap.begin(); iter != labelMap.end(); iter++)
        labels.push_back(iter->first);

    return labels;
}

void FunctionEmitContext::CurrentLanesReturned(Expr *expr, bool doCoherenceCheck) {
    const Type *returnType = function->GetReturnType();
    if (returnType->IsVoidType()) {
        if (expr != NULL)
            Error(expr->pos, "Can't return non-void type \"%s\" from void function.",
                  expr->GetType()->GetString().c_str());
    } else {
        if (expr == NULL) {
            Error(funcStartPos, "Must provide return value for return "
                                "statement for non-void function.");
            return;
        }

        expr = TypeConvertExpr(expr, returnType, "return statement");
        if (expr != NULL) {
            llvm::Value *retVal = expr->GetValue(this);
            if (retVal != NULL) {
                if (returnType->IsUniformType() || CastType<ReferenceType>(returnType) != NULL)
                    StoreInst(retVal, returnValuePtr, returnType, returnType->IsUniformType());
                else {
                    // Use a masked store to store the value of the expression
                    // in the return value memory; this preserves the return
                    // values from other lanes that may have executed return
                    // statements previously.
                    StoreInst(retVal, returnValuePtr, GetInternalMask(), returnType,
                              PointerType::GetUniform(returnType));
                }
            }
        }
    }

    if (emitXeHardwareMask() || (!emitXeHardwareMask() && VaryingCFDepth() == 0)) {
        // Don't need to create mask management instructions for Xe
        // since execution is managed through Xe EM

        // If there is only uniform control flow between us and the
        // function entry, then it's guaranteed that all lanes are running,
        // so we can just emit a true return instruction
        AddInstrumentationPoint("return: uniform control flow");
        ReturnInst();
    } else {
        // Otherwise we update the returnedLanes value by ANDing it with
        // the current lane mask.
        llvm::Value *oldReturnedLanes = LoadInst(returnedLanesPtr, NULL, "old_returned_lanes");
        llvm::Value *newReturnedLanes =
            BinaryOperator(llvm::Instruction::Or, oldReturnedLanes, GetFullMask(), "old_mask|returned_lanes");

        // For 'coherent' return statements, emit code to check if all
        // lanes have returned
        if (doCoherenceCheck) {
            // if newReturnedLanes == functionMaskValue, get out of here!
            llvm::Value *cmp = MasksAllEqual(GetFunctionMask(), newReturnedLanes);
            llvm::BasicBlock *bDoReturn = CreateBasicBlock("do_return");
            llvm::BasicBlock *bNoReturn = CreateBasicBlock("no_return");
            BranchInst(bDoReturn, bNoReturn, cmp);

            bblock = bDoReturn;
            AddInstrumentationPoint("return: all lanes have returned");
            ReturnInst();

            bblock = bNoReturn;
        }
        // Otherwise update returnedLanesPtr and turn off all of the lanes
        // in the current mask so that any subsequent statements in the
        // same scope after the return have no effect
        StoreInst(newReturnedLanes, returnedLanesPtr);
        AddInstrumentationPoint("return: some but not all lanes have returned");
        SetInternalMask(LLVMMaskAllOff);
    }
}

llvm::Value *FunctionEmitContext::Any(llvm::Value *mask) {
    // Call the target-dependent any function to test that the mask is non-zero

    std::vector<Symbol *> mm;
    m->symbolTable->LookupFunction("__any", &mm);
    if (g->target->getMaskBitCount() == 1)
        AssertPos(currentPos, mm.size() == 1);
    else
        // There should be one with signed int signature, one unsigned int.
        AssertPos(currentPos, mm.size() == 2);
    // We can actually call either one, since both are i32s as far as
    // LLVM's type system is concerned...
    llvm::Function *fmm = mm[0]->function;
    return CallInst(fmm, NULL, mask, llvm::Twine(mask->getName()) + "_any");
}

llvm::Value *FunctionEmitContext::All(llvm::Value *mask) {
    // Call the target-dependent movmsk function to turn the vector mask
    // into an i64 value
    std::vector<Symbol *> mm;
    m->symbolTable->LookupFunction("__all", &mm);
    if (g->target->getMaskBitCount() == 1)
        AssertPos(currentPos, mm.size() == 1);
    else
        // There should be one with signed int signature, one unsigned int.
        AssertPos(currentPos, mm.size() == 2);
    // We can actually call either one, since both are i32s as far as
    // LLVM's type system is concerned...
    llvm::Function *fmm = mm[0]->function;
    return CallInst(fmm, NULL, mask, llvm::Twine(mask->getName()) + "_all");
}

llvm::Value *FunctionEmitContext::None(llvm::Value *mask) {
    // Call the target-dependent movmsk function to turn the vector mask
    // into an i64 value
    std::vector<Symbol *> mm;
    m->symbolTable->LookupFunction("__none", &mm);
    if (g->target->getMaskBitCount() == 1)
        AssertPos(currentPos, mm.size() == 1);
    else
        // There should be one with signed int signature, one unsigned int.
        AssertPos(currentPos, mm.size() == 2);
    // We can actually call either one, since both are i32s as far as
    // LLVM's type system is concerned...
    llvm::Function *fmm = mm[0]->function;
    return CallInst(fmm, NULL, mask, llvm::Twine(mask->getName()) + "_none");
}

llvm::Value *FunctionEmitContext::LaneMask(llvm::Value *v) {
    const char *__movmsk = "__movmsk";

    // Call the target-dependent movmsk function to turn the vector mask
    // into an i64 value
    std::vector<Symbol *> mm;
    m->symbolTable->LookupFunction(__movmsk, &mm);
    if (g->target->getMaskBitCount() == 1)
        AssertPos(currentPos, mm.size() == 1);
    else
        // There should be one with signed int signature, one unsigned int.
        AssertPos(currentPos, mm.size() == 2);
    // We can actually call either one, since both are i32s as far as
    // LLVM's type system is concerned...
    llvm::Function *fmm = mm[0]->function;
    return CallInst(fmm, NULL, v, llvm::Twine(v->getName()) + "_movmsk");
}

llvm::Value *FunctionEmitContext::MasksAllEqual(llvm::Value *v1, llvm::Value *v2) {
#if 0
    // Compare the two masks to get a vector of i1s
    llvm::Value *cmp = CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ,
        v1, v2, "v1==v2");
    // Turn that into a bool vector type (often i32s)
    cmp = I1VecToBoolVec(cmp);
    // And see if it's all on
    return All(cmp);
#else
    if (g->target->getArch() == Arch::wasm32) {
        llvm::Function *fmm = m->module->getFunction("__wasm_cmp_msk_eq");
        return CallInst(fmm, NULL, {v1, v2}, ((llvm::Twine("wasm_cmp_msk_eq_") + v1->getName()) + "_") + v2->getName());
    }
    llvm::Value *mm1 = LaneMask(v1);
    llvm::Value *mm2 = LaneMask(v2);
    return CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, mm1, mm2,
                   ((llvm::Twine("equal_") + v1->getName()) + "_") + v2->getName());
#endif
}

llvm::Value *FunctionEmitContext::ProgramIndexVector(bool is32bits) {
    llvm::SmallVector<llvm::Constant *, 16> array;
    for (int i = 0; i < g->target->getVectorWidth(); ++i) {
        llvm::Constant *C = is32bits ? LLVMInt32(i) : LLVMInt64(i);
        array.push_back(C);
    }

    llvm::Constant *index = llvm::ConstantVector::get(array);

    return index;
}

llvm::Value *FunctionEmitContext::GetStringPtr(const std::string &str) {
    llvm::Constant *lstr = llvm::ConstantDataArray::getString(*g->ctx, str);
    llvm::GlobalValue::LinkageTypes linkage = llvm::GlobalValue::InternalLinkage;
    llvm::Value *lstrPtr =
        new llvm::GlobalVariable(*m->module, lstr->getType(), true /*isConst*/, linkage, lstr, "__str");
    return new llvm::BitCastInst(lstrPtr, LLVMTypes::VoidPointerType, "str_void_ptr", bblock);
}

llvm::BasicBlock *FunctionEmitContext::CreateBasicBlock(const llvm::Twine &name, llvm::BasicBlock *insertAfter) {
    llvm::BasicBlock *newBB = llvm::BasicBlock::Create(*g->ctx, name, llvmFunction);
    if (insertAfter)
        newBB->moveAfter(insertAfter);
    return newBB;
}

llvm::Value *FunctionEmitContext::I1VecToBoolVec(llvm::Value *b) {
    if (b == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::ArrayType *at = llvm::dyn_cast<llvm::ArrayType>(b->getType());
    if (at) {
        // If we're given an array of vectors of i1s, then do the
        // conversion for each of the elements.
        // Using 'LLVMTypes::BoolVectorStorageType' since it's a varying short
        // vector and stored as an array in IR.
        llvm::Type *boolArrayType = llvm::ArrayType::get(LLVMTypes::BoolVectorStorageType, at->getNumElements());
        llvm::Value *ret = llvm::UndefValue::get(boolArrayType);

        for (unsigned int i = 0; i < at->getNumElements(); ++i) {
            llvm::Value *elt = ExtractInst(b, i);
            llvm::Value *sext =
                SwitchBoolSize(elt, LLVMTypes::BoolVectorStorageType, llvm::Twine(elt->getName()) + "_to_boolvec");
            ret = InsertInst(ret, sext, i);
        }
        return ret;
    } else {
        // For non-array types, convert to 'LLVMTypes::BoolVectorType' if
        // necessary.
        return SwitchBoolSize(b, LLVMTypes::BoolVectorType, llvm::Twine(b->getName()) + "_to_boolvec");
    }
}

static llvm::Value *lGetStringAsValue(llvm::BasicBlock *bblock, const char *s) {
    llvm::Constant *sConstant = llvm::ConstantDataArray::getString(*g->ctx, s, true);
    std::string var_name = "_";
    var_name = var_name + s;
    llvm::Value *sPtr = new llvm::GlobalVariable(*m->module, sConstant->getType(), true /* const */,
                                                 llvm::GlobalValue::InternalLinkage, sConstant, var_name.c_str());
    llvm::Value *indices[2] = {LLVMInt32(0), LLVMInt32(0)};
    llvm::ArrayRef<llvm::Value *> arrayRef(&indices[0], &indices[2]);
    return llvm::GetElementPtrInst::Create(PTYPE(sPtr), sPtr, arrayRef, "sptr", bblock);
}

void FunctionEmitContext::AddInstrumentationPoint(const char *note) {
    AssertPos(currentPos, note != NULL);
    if (!g->emitInstrumentation)
        return;

    std::vector<llvm::Value *> args;
    // arg 1: filename as string
    args.push_back(lGetStringAsValue(bblock, currentPos.name));
    // arg 2: provided note
    args.push_back(lGetStringAsValue(bblock, note));
    // arg 3: line number
    args.push_back(LLVMInt32(currentPos.first_line));
    // arg 4: current mask, movmsk'ed down to an int64
    args.push_back(LaneMask(GetFullMask()));

    llvm::Function *finst = m->module->getFunction("ISPCInstrument");
    CallInst(finst, NULL, args, "");
}

void FunctionEmitContext::SetDebugPos(SourcePos pos) { currentPos = pos; }

SourcePos FunctionEmitContext::GetDebugPos() const { return currentPos; }

void FunctionEmitContext::AddDebugPos(llvm::Value *value, const SourcePos *pos, llvm::DIScope *scope) {
    llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(value);
    if (inst != NULL && m->diBuilder) {
        SourcePos p = pos ? *pos : currentPos;
        if (p.first_line != 0) {
            // If first_line == 0, then we're in the middle of setting up
            // the standard library or the like; don't add debug positions
            // for those functions
            scope = scope ? scope : GetDIScope();
            llvm::DebugLoc diLoc =
                llvm::DILocation::get(scope->getContext(), p.first_line, p.first_column, scope, nullptr, false);
            inst->setDebugLoc(diLoc);
        }
    }
}

void FunctionEmitContext::StartScope() {
    if (m->diBuilder != NULL) {
        llvm::DIScope *parentScope;
        llvm::DILexicalBlock *lexicalBlock;
        if (debugScopes.size() > 0)
            parentScope = debugScopes.back();
        else
            parentScope = diSubprogram;
        lexicalBlock = m->diBuilder->createLexicalBlock(parentScope, diFile, currentPos.first_line,
                                                        // Revision 216239 in LLVM removes support of DWARF
                                                        // discriminator as the last argument
                                                        currentPos.first_column);
        debugScopes.push_back(llvm::cast<llvm::DILexicalBlockBase>(lexicalBlock));
    }
}

void FunctionEmitContext::EndScope() {
    if (m->diBuilder != NULL) {
        AssertPos(currentPos, debugScopes.size() > 0);
        debugScopes.pop_back();
    }
}

llvm::DIScope *FunctionEmitContext::GetDIScope() const {
    AssertPos(currentPos, debugScopes.size() > 0);
    return debugScopes.back();
}

void FunctionEmitContext::EmitVariableDebugInfo(Symbol *sym) {
    if (m->diBuilder == NULL)
        return;

    llvm::DIScope *scope = GetDIScope();
    llvm::DIType *diType = sym->type->GetDIType(scope);
    llvm::DILocalVariable *var = m->diBuilder->createAutoVariable(
        scope, sym->name, sym->pos.GetDIFile(), sym->pos.first_line, diType, true /* preserve through opts */);

    llvm::DebugLoc diLoc =
        llvm::DILocation::get(scope->getContext(), sym->pos.first_line, sym->pos.first_column, scope, nullptr, false);
    llvm::Instruction *declareInst =
        m->diBuilder->insertDeclare(sym->storagePtr, var, m->diBuilder->createExpression(), diLoc, bblock);
    AddDebugPos(declareInst, &sym->pos, scope);
}

void FunctionEmitContext::EmitFunctionParameterDebugInfo(Symbol *sym, int argNum) {
    if (m->diBuilder == NULL)
        return;

    llvm::DINode::DIFlags flags = llvm::DINode::FlagZero;
    llvm::DIScope *scope = diSubprogram;
    llvm::DIType *diType = sym->type->GetDIType(scope);
    llvm::DILocalVariable *var =
        m->diBuilder->createParameterVariable(scope, sym->name, argNum + 1, sym->pos.GetDIFile(), sym->pos.first_line,
                                              diType, true /* preserve through opts */, flags);

    llvm::DebugLoc diLoc =
        llvm::DILocation::get(scope->getContext(), sym->pos.first_line, sym->pos.first_column, scope, nullptr, false);
    llvm::Instruction *declareInst =
        m->diBuilder->insertDeclare(sym->storagePtr, var, m->diBuilder->createExpression(), diLoc, bblock);
    AddDebugPos(declareInst, &sym->pos, scope);
}

/** If the given type is an array of vector types, then it's the
    representation of an ispc VectorType with varying elements.  If it is
    one of these, return the array size (i.e. the VectorType's size).
    Otherwise return zero.
 */
static int lArrayVectorWidth(llvm::Type *t) {
    llvm::ArrayType *arrayType = llvm::dyn_cast<llvm::ArrayType>(t);
    if (arrayType == NULL) {
        return 0;
    }

    // We shouldn't be seeing arrays of anything but vectors being passed
    // to things like FunctionEmitContext::BinaryOperator() as operands.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    llvm::FixedVectorType *vectorElementType = llvm::dyn_cast<llvm::FixedVectorType>(arrayType->getElementType());
#else
    llvm::VectorType *vectorElementType = llvm::dyn_cast<llvm::VectorType>(arrayType->getElementType());
#endif
    Assert((vectorElementType != NULL && (int)vectorElementType->getNumElements() == g->target->getVectorWidth()));

    return (int)arrayType->getNumElements();
}

llvm::Value *FunctionEmitContext::BinaryOperator(llvm::Instruction::BinaryOps inst, llvm::Value *v0, llvm::Value *v1,
                                                 const llvm::Twine &name) {
    if (v0 == NULL || v1 == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    AssertPos(currentPos, v0->getType() == v1->getType());
    llvm::Type *type = v0->getType();
    int arraySize = lArrayVectorWidth(type);
    if (arraySize == 0) {
        llvm::Instruction *bop = llvm::BinaryOperator::Create(inst, v0, v1, name, bblock);
        AddDebugPos(bop);
        return bop;
    } else {
        // If this is an ispc VectorType, apply the binary operator to each
        // of the elements of the array (which in turn should be either
        // scalar types or llvm::VectorTypes.)
        llvm::Value *ret = llvm::UndefValue::get(type);
        for (int i = 0; i < arraySize; ++i) {
            llvm::Value *a = ExtractInst(v0, i);
            llvm::Value *b = ExtractInst(v1, i);
            llvm::Value *op = BinaryOperator(inst, a, b);
            ret = InsertInst(ret, op, i);
        }
        return ret;
    }
}

llvm::Value *FunctionEmitContext::NotOperator(llvm::Value *v, const llvm::Twine &name) {
    if (v == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    // Similarly to BinaryOperator, do the operation on all the elements of
    // the array if we're given an array type; otherwise just do the
    // regular llvm operation.
    llvm::Type *type = v->getType();
    int arraySize = lArrayVectorWidth(type);
    if (arraySize == 0) {
        llvm::Instruction *binst = llvm::BinaryOperator::CreateNot(v, name.isTriviallyEmpty() ? "not" : name, bblock);
        AddDebugPos(binst);
        return binst;
    } else {
        llvm::Value *ret = llvm::UndefValue::get(type);
        for (int i = 0; i < arraySize; ++i) {
            llvm::Value *a = ExtractInst(v, i);
            llvm::Value *op = llvm::BinaryOperator::CreateNot(a, name.isTriviallyEmpty() ? "not" : name, bblock);
            AddDebugPos(op);
            ret = InsertInst(ret, op, i);
        }
        return ret;
    }
}

// Given the llvm Type that represents an ispc VectorType, return an
// equally-shaped type with boolean elements.  (This is the type that will
// be returned from CmpInst with ispc VectorTypes).
static llvm::Type *lGetMatchingBoolVectorType(llvm::Type *type) {
    llvm::ArrayType *arrayType = llvm::dyn_cast<llvm::ArrayType>(type);
    Assert(arrayType != NULL);

#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    llvm::FixedVectorType *vectorElementType = llvm::dyn_cast<llvm::FixedVectorType>(arrayType->getElementType());
#else
    llvm::VectorType *vectorElementType = llvm::dyn_cast<llvm::VectorType>(arrayType->getElementType());
#endif
    Assert(vectorElementType != NULL);
    Assert((int)vectorElementType->getNumElements() == g->target->getVectorWidth());

    llvm::Type *base = LLVMVECTOR::get(LLVMTypes::BoolType, g->target->getVectorWidth());
    return llvm::ArrayType::get(base, arrayType->getNumElements());
}

llvm::Value *FunctionEmitContext::CmpInst(llvm::Instruction::OtherOps inst, llvm::CmpInst::Predicate pred,
                                          llvm::Value *v0, llvm::Value *v1, const llvm::Twine &name) {
    if (v0 == NULL || v1 == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    AssertPos(currentPos, v0->getType() == v1->getType());
    llvm::Type *type = v0->getType();
    int arraySize = lArrayVectorWidth(type);
    if (arraySize == 0) {
        llvm::Instruction *ci =
            llvm::CmpInst::Create(inst, pred, v0, v1, name.isTriviallyEmpty() ? "cmp" : name, bblock);
        AddDebugPos(ci);
        return ci;
    } else {
        llvm::Type *boolType = lGetMatchingBoolVectorType(type);
        llvm::Value *ret = llvm::UndefValue::get(boolType);
        for (int i = 0; i < arraySize; ++i) {
            llvm::Value *a = ExtractInst(v0, i);
            llvm::Value *b = ExtractInst(v1, i);
            llvm::Value *op = CmpInst(inst, pred, a, b, name.isTriviallyEmpty() ? "cmp" : name);
            ret = InsertInst(ret, op, i);
        }
        return ret;
    }
}

llvm::Value *FunctionEmitContext::SmearUniform(llvm::Value *value, const llvm::Twine &name) {
    if (value == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::Value *ret = NULL;
    llvm::Type *eltType = value->getType();
    llvm::Type *vecType = NULL;

    llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(eltType);
    if (pt != NULL) {
        // Varying pointers are represented as vectors of i32/i64s
        vecType = LLVMTypes::VoidPointerVectorType;
        value = PtrToIntInst(value);
    } else {
        // All other varying types are represented as vectors of the
        // underlying type.
        vecType = LLVMVECTOR::get(eltType, g->target->getVectorWidth());
    }

    // Check for a constant case.
    if (llvm::Constant *const_val = llvm::dyn_cast<llvm::Constant>(value)) {
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
        ret = llvm::ConstantVector::getSplat(g->target->getVectorWidth(), const_val);
#elif ISPC_LLVM_VERSION < ISPC_LLVM_12_0
        ret =
            llvm::ConstantVector::getSplat({static_cast<unsigned int>(g->target->getVectorWidth()), false}, const_val);
#else // LLVM 12.0+
        ret = llvm::ConstantVector::getSplat(
            llvm::ElementCount::get(static_cast<unsigned int>(g->target->getVectorWidth()), false), const_val);
#endif
        return ret;
    }

    ret = BroadcastValue(value, vecType, name);

    return ret;
}

llvm::Value *FunctionEmitContext::BitCastInst(llvm::Value *value, llvm::Type *type, const llvm::Twine &name) {
    if (value == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::Instruction *inst = new llvm::BitCastInst(
        value, type, name.isTriviallyEmpty() ? llvm::Twine(value->getName()) + "_bitcast" : name, bblock);
    AddDebugPos(inst);
    return inst;
}

llvm::Value *FunctionEmitContext::PtrToIntInst(llvm::Value *value, const llvm::Twine &name) {
    if (value == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    if (llvm::isa<llvm::VectorType>(value->getType()))
        // no-op for varying pointers; they're already vectors of ints
        return value;

    llvm::Type *type = LLVMTypes::PointerIntType;
    llvm::Instruction *inst = new llvm::PtrToIntInst(
        value, type, name.isTriviallyEmpty() ? llvm::Twine(value->getName()) + "_ptr2int" : name, bblock);
    AddDebugPos(inst);
    return inst;
}

llvm::Value *FunctionEmitContext::PtrToIntInst(llvm::Value *value, llvm::Type *toType, const llvm::Twine &name) {
    if (value == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::Type *fromType = value->getType();
    if (llvm::isa<llvm::VectorType>(fromType)) {
        // varying pointer
        if (fromType == toType)
            // already the right type--done
            return value;
        else if (fromType->getScalarSizeInBits() > toType->getScalarSizeInBits())
            return TruncInst(value, toType,
                             name.isTriviallyEmpty() ? llvm::Twine(value->getName()) + "_ptr2int" : name);
        else {
            AssertPos(currentPos, fromType->getScalarSizeInBits() < toType->getScalarSizeInBits());
            return ZExtInst(value, toType, name.isTriviallyEmpty() ? llvm::Twine(value->getName()) + "_ptr2int" : name);
        }
    }

    llvm::Instruction *inst = new llvm::PtrToIntInst(
        value, toType, name.isTriviallyEmpty() ? llvm::Twine(value->getName()) + "_ptr2int" : name, bblock);
    AddDebugPos(inst);
    return inst;
}

llvm::Value *FunctionEmitContext::IntToPtrInst(llvm::Value *value, llvm::Type *toType, const llvm::Twine &name) {
    if (value == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::Type *fromType = value->getType();
    if (llvm::isa<llvm::VectorType>(fromType)) {
        // varying pointer
        if (fromType == toType)
            // done
            return value;
        else if (fromType->getScalarSizeInBits() > toType->getScalarSizeInBits())
            return TruncInst(value, toType,
                             name.isTriviallyEmpty() ? llvm::Twine(value->getName()) + "_int2ptr" : name);
        else {
            AssertPos(currentPos, fromType->getScalarSizeInBits() < toType->getScalarSizeInBits());
            return ZExtInst(value, toType, name.isTriviallyEmpty() ? llvm::Twine(value->getName()) + "_int2ptr" : name);
        }
    }

    llvm::Instruction *inst = new llvm::IntToPtrInst(
        value, toType, name.isTriviallyEmpty() ? llvm::Twine(value->getName()) + "_int2ptr" : name, bblock);
    AddDebugPos(inst);
    return inst;
}

llvm::Instruction *FunctionEmitContext::TruncInst(llvm::Value *value, llvm::Type *type, const llvm::Twine &name) {
    if (value == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    // TODO: we should probably handle the array case as in
    // e.g. BitCastInst(), but we don't currently need that functionality
    llvm::Instruction *inst = new llvm::TruncInst(
        value, type, name.isTriviallyEmpty() ? llvm::Twine(value->getName()) + "_trunc" : name, bblock);
    AddDebugPos(inst);
    return inst;
}

llvm::Instruction *FunctionEmitContext::CastInst(llvm::Instruction::CastOps op, llvm::Value *value, llvm::Type *type,
                                                 const llvm::Twine &name) {
    if (value == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    // TODO: we should probably handle the array case as in
    // e.g. BitCastInst(), but we don't currently need that functionality
    llvm::Instruction *inst = llvm::CastInst::Create(
        op, value, type, name.isTriviallyEmpty() ? llvm::Twine(value->getName()) + "_cast" : name, bblock);
    AddDebugPos(inst);
    return inst;
}

llvm::Instruction *FunctionEmitContext::FPCastInst(llvm::Value *value, llvm::Type *type, const llvm::Twine &name) {
    if (value == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    // TODO: we should probably handle the array case as in
    // e.g. BitCastInst(), but we don't currently need that functionality
    llvm::Instruction *inst = llvm::CastInst::CreateFPCast(
        value, type, name.isTriviallyEmpty() ? llvm::Twine(value->getName()) + "_cast" : name, bblock);
    AddDebugPos(inst);
    return inst;
}

llvm::Instruction *FunctionEmitContext::SExtInst(llvm::Value *value, llvm::Type *type, const llvm::Twine &name) {
    if (value == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    // TODO: we should probably handle the array case as in
    // e.g. BitCastInst(), but we don't currently need that functionality
    llvm::Instruction *inst = new llvm::SExtInst(
        value, type, name.isTriviallyEmpty() ? llvm::Twine(value->getName()) + "_sext" : name, bblock);
    AddDebugPos(inst);
    return inst;
}

llvm::Instruction *FunctionEmitContext::ZExtInst(llvm::Value *value, llvm::Type *type, const llvm::Twine &name) {
    if (value == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    // TODO: we should probably handle the array case as in
    // e.g. BitCastInst(), but we don't currently need that functionality
    llvm::Instruction *inst = new llvm::ZExtInst(
        value, type, name.isTriviallyEmpty() ? llvm::Twine(value->getName()) + "_zext" : name, bblock);
    AddDebugPos(inst);
    return inst;
}

/** Utility routine used by the GetElementPtrInst() methods; given a
    pointer to some type (either uniform or varying) and an index (also
    either uniform or varying), this returns the new pointer (varying if
    appropriate) given by offsetting the base pointer by the index times
    the size of the object that the pointer points to.
 */
llvm::Value *FunctionEmitContext::applyVaryingGEP(llvm::Value *basePtr, llvm::Value *index, const Type *ptrType) {
    // Find the scale factor for the index (i.e. the size of the object
    // that the pointer(s) point(s) to.
    const Type *scaleType = ptrType->GetBaseType();
    llvm::Value *scale = g->target->SizeOf(scaleType->LLVMType(g->ctx), bblock);

    bool indexIsVarying = llvm::isa<llvm::VectorType>(index->getType());
    llvm::Value *offset = NULL;
    if (indexIsVarying == false) {
        // Truncate or sign extend the index as appropriate to a 32 or
        // 64-bit type.
        if ((g->target->is32Bit() || g->opt.force32BitAddressing) && index->getType() == LLVMTypes::Int64Type)
            index = TruncInst(index, LLVMTypes::Int32Type);
        else if ((!g->target->is32Bit() && !g->opt.force32BitAddressing) && index->getType() == LLVMTypes::Int32Type)
            index = SExtInst(index, LLVMTypes::Int64Type);

        // do a scalar multiply to get the offset as index * scale and then
        // smear the result out to be a vector; this is more efficient than
        // first promoting both the scale and the index to vectors and then
        // multiplying.
        offset = BinaryOperator(llvm::Instruction::Mul, scale, index);
        offset = SmearUniform(offset);
    } else {
        // Similarly, truncate or sign extend the index to be a 32 or 64
        // bit vector type
        if ((g->target->is32Bit() || g->opt.force32BitAddressing) && index->getType() == LLVMTypes::Int64VectorType)
            index = TruncInst(index, LLVMTypes::Int32VectorType);
        else if ((!g->target->is32Bit() && !g->opt.force32BitAddressing) &&
                 index->getType() == LLVMTypes::Int32VectorType)
            index = SExtInst(index, LLVMTypes::Int64VectorType);

        scale = SmearUniform(scale);
        Assert(index != NULL);
        // offset = index * scale
        offset = BinaryOperator(llvm::Instruction::Mul, scale, index,
                                ((llvm::Twine("mul_") + scale->getName()) + "_") + index->getName());
    }

    // For 64-bit targets, if we've been doing our offset calculations in
    // 32 bits, we still have to convert to a 64-bit value before we
    // actually add the offset to the pointer.
    if (g->target->is32Bit() == false && g->opt.force32BitAddressing == true)
        offset = SExtInst(offset, LLVMTypes::Int64VectorType, llvm::Twine(offset->getName()) + "_to_64");

    // Smear out the pointer to be varying; either the base pointer or the
    // index must be varying for this method to be called.
    bool baseIsUniform = (llvm::isa<llvm::PointerType>(basePtr->getType()));
    AssertPos(currentPos, baseIsUniform == false || indexIsVarying == true);
    llvm::Value *varyingPtr = baseIsUniform ? SmearUniform(basePtr) : basePtr;

    // newPtr = ptr + offset
    return BinaryOperator(llvm::Instruction::Add, varyingPtr, offset, llvm::Twine(basePtr->getName()) + "_offset");
}

void FunctionEmitContext::MatchIntegerTypes(llvm::Value **v0, llvm::Value **v1) {
    llvm::Type *type0 = (*v0)->getType();
    llvm::Type *type1 = (*v1)->getType();

    // First, promote to a vector type if one of the two values is a vector
    // type
    if (llvm::isa<llvm::VectorType>(type0) && !llvm::isa<llvm::VectorType>(type1)) {
        *v1 = SmearUniform(*v1, "smear_v1");
        type1 = (*v1)->getType();
    }
    if (!llvm::isa<llvm::VectorType>(type0) && llvm::isa<llvm::VectorType>(type1)) {
        *v0 = SmearUniform(*v0, "smear_v0");
        type0 = (*v0)->getType();
    }

    // And then update to match bit widths
    if (type0 == LLVMTypes::Int32Type && type1 == LLVMTypes::Int64Type)
        *v0 = SExtInst(*v0, LLVMTypes::Int64Type);
    else if (type1 == LLVMTypes::Int32Type && type0 == LLVMTypes::Int64Type)
        *v1 = SExtInst(*v1, LLVMTypes::Int64Type);
    else if (type0 == LLVMTypes::Int32VectorType && type1 == LLVMTypes::Int64VectorType)
        *v0 = SExtInst(*v0, LLVMTypes::Int64VectorType);
    else if (type1 == LLVMTypes::Int32VectorType && type0 == LLVMTypes::Int64VectorType)
        *v1 = SExtInst(*v1, LLVMTypes::Int64VectorType);
}

/** Given an integer index in indexValue that's indexing into an array of
    soa<> structures with given soaWidth, compute the two sub-indices we
    need to do the actual indexing calculation:

    subIndices[0] = (indexValue >> log(soaWidth))
    subIndices[1] = (indexValue & (soaWidth-1))
 */
static llvm::Value *lComputeSliceIndex(FunctionEmitContext *ctx, int soaWidth, llvm::Value *indexValue,
                                       llvm::Value *ptrSliceOffset, llvm::Value **newSliceOffset) {
    // Compute the log2 of the soaWidth.
    Assert(soaWidth > 0);
    int logWidth = 0, sw = soaWidth;
    while (sw > 1) {
        ++logWidth;
        sw >>= 1;
    }
    Assert((1 << logWidth) == soaWidth);

    ctx->MatchIntegerTypes(&indexValue, &ptrSliceOffset);
    Assert(indexValue != NULL);
    llvm::Type *indexType = indexValue->getType();
    llvm::Value *shift = LLVMIntAsType(logWidth, indexType);
    llvm::Value *mask = LLVMIntAsType(soaWidth - 1, indexType);

    llvm::Value *indexSum = ctx->BinaryOperator(llvm::Instruction::Add, indexValue, ptrSliceOffset, "index_sum");

    // minor index = (index & (soaWidth - 1))
    *newSliceOffset = ctx->BinaryOperator(llvm::Instruction::And, indexSum, mask, "slice_index_minor");
    // slice offsets are always 32 bits...
    if ((*newSliceOffset)->getType() == LLVMTypes::Int64Type)
        *newSliceOffset = ctx->TruncInst(*newSliceOffset, LLVMTypes::Int32Type);
    else if ((*newSliceOffset)->getType() == LLVMTypes::Int64VectorType)
        *newSliceOffset = ctx->TruncInst(*newSliceOffset, LLVMTypes::Int32VectorType);

    // major index = (index >> logWidth)
    return ctx->BinaryOperator(llvm::Instruction::AShr, indexSum, shift, "slice_index_major");
}

llvm::Value *FunctionEmitContext::MakeSlicePointer(llvm::Value *ptr, llvm::Value *offset) {
    // Create a small struct where the first element is the type of the
    // given pointer and the second element is the type of the offset
    // value.
    std::vector<llvm::Type *> eltTypes;
    eltTypes.push_back(ptr->getType());
    eltTypes.push_back(offset->getType());
    llvm::StructType *st = llvm::StructType::get(*g->ctx, eltTypes);

    llvm::Value *ret = llvm::UndefValue::get(st);
    ret = InsertInst(ret, ptr, 0, llvm::Twine(ret->getName()) + "_slice_ptr");
    ret = InsertInst(ret, offset, 1, llvm::Twine(ret->getName()) + "_slice_offset");
    return ret;
}

llvm::Value *FunctionEmitContext::GetElementPtrInst(llvm::Value *basePtr, llvm::Value *index, const Type *ptrRefType,
                                                    const llvm::Twine &name) {
    if (basePtr == NULL || index == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    // Regularize to a standard pointer type for basePtr's type
    const PointerType *ptrType;
    if (CastType<ReferenceType>(ptrRefType) != NULL)
        ptrType = PointerType::GetUniform(ptrRefType->GetReferenceTarget());
    else {
        ptrType = CastType<PointerType>(ptrRefType);
    }
    AssertPos(currentPos, ptrType != NULL);

    if (ptrType->IsSlice()) {
        AssertPos(currentPos, llvm::isa<llvm::StructType>(basePtr->getType()));

        llvm::Value *ptrSliceOffset = ExtractInst(basePtr, 1);
        if (ptrType->IsFrozenSlice() == false) {
            // For slice pointers that aren't frozen, we compute a new
            // index based on the given index plus the offset in the slice
            // pointer.  This gives us an updated integer slice index for
            // the resulting slice pointer and then an index to index into
            // the soa<> structs with.
            llvm::Value *newSliceOffset;
            int soaWidth = ptrType->GetBaseType()->GetSOAWidth();
            index = lComputeSliceIndex(this, soaWidth, index, ptrSliceOffset, &newSliceOffset);
            ptrSliceOffset = newSliceOffset;
        }

        // Handle the indexing into the soa<> structs with the major
        // component of the index through a recursive call
        llvm::Value *p = GetElementPtrInst(ExtractInst(basePtr, 0), index, ptrType->GetAsNonSlice(), name);

        // And mash the results together for the return value
        return MakeSlicePointer(p, ptrSliceOffset);
    }

    // Double-check consistency between the given pointer type and its LLVM
    // type.
    if (ptrType->IsUniformType())
        AssertPos(currentPos, llvm::isa<llvm::PointerType>(basePtr->getType()));
    else if (ptrType->IsVaryingType())
        AssertPos(currentPos, llvm::isa<llvm::VectorType>(basePtr->getType()));

    bool indexIsVaryingType = llvm::isa<llvm::VectorType>(index->getType());

    if (indexIsVaryingType == false && ptrType->IsUniformType() == true) {
        // The easy case: both the base pointer and the indices are
        // uniform, so just emit the regular LLVM GEP instruction
        llvm::Value *ind[1] = {index};
        llvm::ArrayRef<llvm::Value *> arrayRef(&ind[0], &ind[1]);
        llvm::Instruction *inst = llvm::GetElementPtrInst::Create(PTYPE(basePtr), basePtr, arrayRef,
                                                                  name.isTriviallyEmpty() ? "gep" : name, bblock);
        AddDebugPos(inst);
        return inst;
    } else
        return applyVaryingGEP(basePtr, index, ptrType);
}

llvm::Value *FunctionEmitContext::GetElementPtrInst(llvm::Value *basePtr, llvm::Value *index0, llvm::Value *index1,
                                                    const Type *ptrRefType, const llvm::Twine &name) {
    if (basePtr == NULL || index0 == NULL || index1 == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    // Regaularize the pointer type for basePtr
    const PointerType *ptrType = NULL;
    if (CastType<ReferenceType>(ptrRefType) != NULL)
        ptrType = PointerType::GetUniform(ptrRefType->GetReferenceTarget());
    else {
        ptrType = CastType<PointerType>(ptrRefType);
        AssertPos(currentPos, ptrType != NULL);
    }

    if (ptrType->IsSlice()) {
        // Similar to the 1D GEP implementation above, for non-frozen slice
        // pointers we do the two-step indexing calculation and then pass
        // the new major index on to a recursive GEP call.
        AssertPos(currentPos, llvm::isa<llvm::StructType>(basePtr->getType()));
        llvm::Value *ptrSliceOffset = ExtractInst(basePtr, 1);
        if (ptrType->IsFrozenSlice() == false) {
            llvm::Value *newSliceOffset;
            int soaWidth = ptrType->GetBaseType()->GetSOAWidth();
            index1 = lComputeSliceIndex(this, soaWidth, index1, ptrSliceOffset, &newSliceOffset);
            ptrSliceOffset = newSliceOffset;
        }

        llvm::Value *p = GetElementPtrInst(ExtractInst(basePtr, 0), index0, index1, ptrType->GetAsNonSlice(), name);
        return MakeSlicePointer(p, ptrSliceOffset);
    }

    bool index0IsVaryingType = llvm::isa<llvm::VectorType>(index0->getType());
    bool index1IsVaryingType = llvm::isa<llvm::VectorType>(index1->getType());

    if (index0IsVaryingType == false && index1IsVaryingType == false && ptrType->IsUniformType() == true) {
        // The easy case: both the base pointer and the indices are
        // uniform, so just emit the regular LLVM GEP instruction
        llvm::Value *indices[2] = {index0, index1};
        llvm::ArrayRef<llvm::Value *> arrayRef(&indices[0], &indices[2]);
        llvm::Instruction *inst = llvm::GetElementPtrInst::Create(PTYPE(basePtr), basePtr, arrayRef,
                                                                  name.isTriviallyEmpty() ? "gep" : name, bblock);
        AddDebugPos(inst);
        return inst;
    } else {
        // Handle the first dimension with index0
        llvm::Value *ptr0 = GetElementPtrInst(basePtr, index0, ptrType);

        // Now index into the second dimension with index1.  First figure
        // out the type of ptr0.
        const Type *baseType = ptrType->GetBaseType();
        const SequentialType *st = CastType<SequentialType>(baseType);
        AssertPos(currentPos, st != NULL);

        bool ptr0IsUniform = llvm::isa<llvm::PointerType>(ptr0->getType());
        const Type *ptr0BaseType = st->GetElementType();
        const Type *ptr0Type =
            ptr0IsUniform ? PointerType::GetUniform(ptr0BaseType) : PointerType::GetVarying(ptr0BaseType);

        return applyVaryingGEP(ptr0, index1, ptr0Type);
    }
}

llvm::Value *FunctionEmitContext::AddElementOffset(llvm::Value *fullBasePtr, int elementNum, const Type *ptrRefType,
                                                   const llvm::Twine &name, const PointerType **resultPtrType) {
    if (resultPtrType != NULL)
        AssertPos(currentPos, ptrRefType != NULL);

    llvm::PointerType *llvmPtrType = llvm::dyn_cast<llvm::PointerType>(fullBasePtr->getType());
    if (llvmPtrType != NULL) {
        llvm::StructType *llvmStructType = llvm::dyn_cast<llvm::StructType>(llvmPtrType->getElementType());
        if (llvmStructType != NULL && llvmStructType->isSized() == false) {
            AssertPos(currentPos, m->errorCount > 0);
            return NULL;
        }
    }

    // (Unfortunately) it's not required to pass a non-NULL ptrRefType, but
    // if we have one, regularize into a pointer type.
    const PointerType *ptrType = NULL;
    if (ptrRefType != NULL) {
        // Normalize references to uniform pointers
        if (CastType<ReferenceType>(ptrRefType) != NULL)
            ptrType = PointerType::GetUniform(ptrRefType->GetReferenceTarget());
        else
            ptrType = CastType<PointerType>(ptrRefType);
        AssertPos(currentPos, ptrType != NULL);
    }

    // Similarly, we have to see if the pointer type is a struct to see if
    // we have a slice pointer instead of looking at ptrType; this is also
    // unfortunate...
    llvm::Value *basePtr = fullBasePtr;
    bool baseIsSlicePtr = llvm::isa<llvm::StructType>(fullBasePtr->getType());
    const PointerType *rpt;
    if (baseIsSlicePtr) {
        AssertPos(currentPos, ptrType != NULL);
        // Update basePtr to just be the part that actually points to the
        // start of an soa<> struct for now; the element offset computation
        // doesn't change the slice offset, so we'll incorporate that into
        // the final value right before this method returns.
        basePtr = ExtractInst(fullBasePtr, 0);
        if (resultPtrType == NULL)
            resultPtrType = &rpt;
    }

    // Return the pointer type of the result of this call, for callers that
    // want it.
    if (resultPtrType != NULL) {
        AssertPos(currentPos, ptrType != NULL);
        const CollectionType *ct = CastType<CollectionType>(ptrType->GetBaseType());
        AssertPos(currentPos, ct != NULL);
        *resultPtrType = new PointerType(ct->GetElementType(elementNum), ptrType->GetVariability(),
                                         ptrType->IsConstType(), ptrType->IsSlice());
    }

    llvm::Value *resultPtr = NULL;
    if (ptrType == NULL || ptrType->IsUniformType()) {
        // If the pointer is uniform, we can use the regular LLVM GEP.
        llvm::Value *offsets[2] = {LLVMInt32(0), LLVMInt32(elementNum)};
        llvm::ArrayRef<llvm::Value *> arrayRef(&offsets[0], &offsets[2]);
        resultPtr = llvm::GetElementPtrInst::Create(PTYPE(basePtr), basePtr, arrayRef,
                                                    name.isTriviallyEmpty() ? "struct_offset" : name, bblock);
    } else {
        // Otherwise do the math to find the offset and add it to the given
        // varying pointers
        const StructType *st = CastType<StructType>(ptrType->GetBaseType());
        llvm::Value *offset = NULL;
        if (st != NULL)
            // If the pointer is to a structure, Target::StructOffset() gives
            // us the offset in bytes to the given element of the structure
            offset = g->target->StructOffset(st->LLVMType(g->ctx), elementNum, bblock);
        else {
            // Otherwise we should have a vector or array here and the offset
            // is given by the element number times the size of the element
            // type of the vector.
            const SequentialType *st = CastType<SequentialType>(ptrType->GetBaseType());
            AssertPos(currentPos, st != NULL);
            llvm::Value *size = g->target->SizeOf(st->GetElementType()->LLVMType(g->ctx), bblock);
            llvm::Value *scale =
                (g->target->is32Bit() || g->opt.force32BitAddressing) ? LLVMInt32(elementNum) : LLVMInt64(elementNum);
            offset = BinaryOperator(llvm::Instruction::Mul, size, scale);
        }

        offset = SmearUniform(offset, "offset_smear");

        if (g->target->is32Bit() == false && g->opt.force32BitAddressing == true)
            // If we're doing 32 bit addressing with a 64 bit target, although
            // we did the math above in 32 bit, we need to go to 64 bit before
            // we add the offset to the varying pointers.
            offset = SExtInst(offset, LLVMTypes::Int64VectorType, "offset_to_64");

        resultPtr = BinaryOperator(llvm::Instruction::Add, basePtr, offset, "struct_ptr_offset");
    }

    // Finally, if had a slice pointer going in, mash back together with
    // the original (unchanged) slice offset.
    if (baseIsSlicePtr)
        return MakeSlicePointer(resultPtr, ExtractInst(fullBasePtr, 1));
    else
        return resultPtr;
}

llvm::Value *FunctionEmitContext::SwitchBoolSize(llvm::Value *value, llvm::Type *toType, const llvm::Twine &name) {
    if ((value == NULL) || (toType == NULL)) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::Type *fromType = value->getType();

    llvm::Value *newBool = value;
    if (g->target->getDataLayout()->getTypeSizeInBits(fromType) >
        g->target->getDataLayout()->getTypeSizeInBits(toType)) {
        newBool =
            TruncInst(value, toType, name.isTriviallyEmpty() ? (llvm::Twine(value->getName()) + "_switchBool") : name);
    } else if (g->target->getDataLayout()->getTypeSizeInBits(fromType) <
               g->target->getDataLayout()->getTypeSizeInBits(toType)) {
        newBool =
            SExtInst(value, toType, name.isTriviallyEmpty() ? (llvm::Twine(value->getName()) + "_switchBool") : name);
    }

    return newBool;
}

llvm::Value *FunctionEmitContext::LoadInst(llvm::Value *ptr, const Type *type, const llvm::Twine &name) {
    if (ptr == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(ptr->getType());
    AssertPos(currentPos, pt != NULL);

#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    llvm::LoadInst *inst =
        new llvm::LoadInst(pt->getPointerElementType(), ptr,
                           name.isTriviallyEmpty() ? (llvm::Twine(ptr->getName()) + "_load") : name, bblock);
#else
    llvm::LoadInst *inst =
        new llvm::LoadInst(ptr, name.isTriviallyEmpty() ? (llvm::Twine(ptr->getName()) + "_load") : name, bblock);
#endif

    if (g->opt.forceAlignedMemory && llvm::dyn_cast<llvm::VectorType>(pt->getElementType())) {
        inst->setAlignment(llvm::MaybeAlign(g->target->getNativeVectorAlignment()).valueOrOne());
    }

    AddDebugPos(inst);

    llvm::Value *loadVal = inst;
    // bool type is stored as i8. So, it requires some processing.
    if ((type != NULL) && (type->IsBoolType())) {
        if (CastType<AtomicType>(type) != NULL) {
            loadVal = SwitchBoolSize(loadVal, type->LLVMType(g->ctx));
        } else if ((CastType<VectorType>(type) != NULL)) {
            const VectorType *vType = CastType<VectorType>(type);
            if (CastType<AtomicType>(vType->GetElementType()) != NULL) {
                loadVal = SwitchBoolSize(loadVal, type->LLVMType(g->ctx));
            }
        }
    }
    return loadVal;
}

/** Given a slice pointer to soa'd data that is a basic type (atomic,
    pointer, or enum type), use the slice offset to compute pointer(s) to
    the appropriate individual data element(s).
 */
static llvm::Value *lFinalSliceOffset(FunctionEmitContext *ctx, llvm::Value *ptr, const PointerType **ptrType) {
    Assert(CastType<PointerType>(*ptrType) != NULL);

    llvm::Value *slicePtr = ctx->ExtractInst(ptr, 0, llvm::Twine(ptr->getName()) + "_ptr");
    llvm::Value *sliceOffset = ctx->ExtractInst(ptr, 1, llvm::Twine(ptr->getName()) + "_offset");

    // slicePtr should be a pointer to an soa-width wide array of the
    // final atomic/enum/pointer type
    const Type *unifBaseType = (*ptrType)->GetBaseType()->GetAsUniformType();
    Assert(Type::IsBasicType(unifBaseType));

    // The final pointer type is a uniform or varying pointer to the
    // underlying uniform type, depending on whether the given pointer is
    // uniform or varying.
    *ptrType =
        (*ptrType)->IsUniformType() ? PointerType::GetUniform(unifBaseType) : PointerType::GetVarying(unifBaseType);

    // For uniform pointers, bitcast to a pointer to the uniform element
    // type, so that the GEP below does the desired indexing
    if ((*ptrType)->IsUniformType())
        slicePtr = ctx->BitCastInst(slicePtr, (*ptrType)->LLVMType(g->ctx));

    // And finally index based on the slice offset
    return ctx->GetElementPtrInst(slicePtr, sliceOffset, *ptrType, llvm::Twine(slicePtr->getName()) + "_final_gep");
}

/** Utility routine that loads from a uniform pointer to soa<> data,
    returning a regular uniform (non-SOA result).
 */
llvm::Value *FunctionEmitContext::loadUniformFromSOA(llvm::Value *ptr, llvm::Value *mask, const PointerType *ptrType,
                                                     const llvm::Twine &name) {
    const Type *unifType = ptrType->GetBaseType()->GetAsUniformType();

    const CollectionType *ct = CastType<CollectionType>(ptrType->GetBaseType());
    if (ct != NULL) {
        // If we have a struct/array, we need to decompose it into
        // individual element loads to fill in the result structure since
        // the SOA slice of values we need isn't contiguous in memory...
        llvm::Type *llvmReturnType = unifType->LLVMType(g->ctx);
        llvm::Value *retValue = llvm::UndefValue::get(llvmReturnType);

        for (int i = 0; i < ct->GetElementCount(); ++i) {
            const PointerType *eltPtrType;
            llvm::Value *eltPtr = AddElementOffset(ptr, i, ptrType, "elt_offset", &eltPtrType);
            llvm::Value *eltValue = LoadInst(eltPtr, mask, eltPtrType, name);
            retValue = InsertInst(retValue, eltValue, i, "set_value");
        }

        return retValue;
    } else {
        // Otherwise we've made our way to a slice pointer to a basic type;
        // we need to apply the slice offset into this terminal SOA array
        // and then perform the final load
        ptr = lFinalSliceOffset(this, ptr, &ptrType);
        return LoadInst(ptr, mask, ptrType, name);
    }
}

llvm::Value *FunctionEmitContext::LoadInst(llvm::Value *ptr, llvm::Value *mask, const Type *ptrRefType,
                                           const llvm::Twine &name, bool one_elem) {
    if (ptr == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    AssertPos(currentPos, ptrRefType != NULL && mask != NULL);

    const PointerType *ptrType;
    const Type *elType;
    if (CastType<ReferenceType>(ptrRefType) != NULL) {
        ptrType = PointerType::GetUniform(ptrRefType->GetReferenceTarget());
        elType = ptrRefType->GetReferenceTarget();
    } else {
        ptrType = CastType<PointerType>(ptrRefType);
        AssertPos(currentPos, ptrType != NULL);
        elType = ptrType->GetBaseType();
    }

    if (CastType<UndefinedStructType>(ptrType->GetBaseType())) {
        Error(currentPos, "Unable to load to undefined struct type \"%s\".",
              ptrType->GetBaseType()->GetString().c_str());
        return NULL;
    }

    if (ptrType->IsUniformType()) {
        if (ptrType->IsSlice()) {
            return loadUniformFromSOA(ptr, mask, ptrType,
                                      name.isTriviallyEmpty() ? (llvm::Twine(ptr->getName()) + "_load") : name);
        } else {
            // FIXME: same issue as above load inst regarding alignment...
            //
            // If the ptr is a straight up regular pointer, then just issue
            // a regular load.  First figure out the alignment; in general we
            // can just assume the natural alignment (0 here), but for varying
            // atomic types, we need to make sure that the compiler emits
            // unaligned vector loads, so we specify a reduced alignment here.
            const AtomicType *atomicType = CastType<AtomicType>(ptrType->GetBaseType());

#if ISPC_LLVM_VERSION <= ISPC_LLVM_10_0
            llvm::LoadInst *inst =
                new llvm::LoadInst(ptr, name.isTriviallyEmpty() ? (llvm::Twine(ptr->getName()) + "_load") : name,
                                   false /* not volatile */, bblock);
#else // LLVM 11.0+
            llvm::PointerType *ptr_type = llvm::dyn_cast<llvm::PointerType>(ptr->getType());
            llvm::LoadInst *inst =
                new llvm::LoadInst(ptr_type->getPointerElementType(), ptr,
                                   name.isTriviallyEmpty() ? (llvm::Twine(ptr->getName()) + "_load") : name,
                                   false /* not volatile */, bblock);
#endif

            if (atomicType != NULL && atomicType->IsVaryingType()) {
                // We actually just want to align to the vector element
                // alignment, but can't easily get that here, so just tell LLVM
                // it's totally unaligned.  (This shouldn't make any difference
                // vs the proper alignment in practice.)
                int align = 1;

                inst->setAlignment(llvm::MaybeAlign(align).valueOrOne());
            }

            AddDebugPos(inst);
            llvm::Value *loadVal = inst;
            // bool type is stored as i8. So, it requires some processing.
            if (elType->IsBoolType() && (CastType<AtomicType>(elType) != NULL)) {
                loadVal = SwitchBoolSize(loadVal, elType->LLVMType(g->ctx));
            }
            return loadVal;
        }
    } else {
        // Otherwise we should have a varying ptr and it's time for a
        // gather.
        llvm::Value *gather_result = gather(ptr, ptrType, GetFullMask(),
                                            name.isTriviallyEmpty() ? (llvm::Twine(ptr->getName()) + "_load") : name);
        if (!one_elem)
            return gather_result;

        // It is a kludge. When we dereference varying pointer to uniform struct
        // with "bound uniform" member, we should return first unmasked member.
        Warning(currentPos, "Dereferencing varying pointer to uniform struct with 'bound uniform' member,\n"
                            " only one value will survive. Possible loss of data.");
        // Call the target-dependent movmsk function to turn the vector mask
        // into an i64 value
        std::vector<Symbol *> mm;
        m->symbolTable->LookupFunction("__movmsk", &mm);
        if (g->target->getMaskBitCount() == 1)
            AssertPos(currentPos, mm.size() == 1);
        else
            // There should be one with signed int signature, one unsigned int.
            AssertPos(currentPos, mm.size() == 2);
        // We can actually call either one, since both are i32s as far as
        // LLVM's type system is concerned...
        llvm::Function *fmm = mm[0]->function;
        llvm::Value *int_mask = CallInst(fmm, NULL, mask, llvm::Twine(mask->getName()) + "_movmsk");
        std::vector<Symbol *> lz;
        m->symbolTable->LookupFunction("__count_trailing_zeros_i64", &lz);
        llvm::Function *flz = lz[0]->function;
        llvm::Value *elem_idx = CallInst(flz, NULL, int_mask, llvm::Twine(mask->getName()) + "_clz");
        llvm::Value *elem = llvm::ExtractElementInst::Create(
            gather_result, elem_idx, llvm::Twine(gather_result->getName()) + "_umasked_elem", bblock);
        return elem;
    }
}

llvm::Value *FunctionEmitContext::gather(llvm::Value *ptr, const PointerType *ptrType, llvm::Value *mask,
                                         const llvm::Twine &name) {
    // We should have a varying pointer if we get here...
    AssertPos(currentPos, ptrType->IsVaryingType());

    const Type *returnType = ptrType->GetBaseType()->GetAsVaryingType();
    llvm::Type *llvmReturnType = returnType->LLVMType(g->ctx);
    const CollectionType *collectionType = CastType<CollectionType>(ptrType->GetBaseType());
    if (collectionType != NULL) {
        // For collections, recursively gather element wise to find the
        // result.
        llvm::Value *retValue = llvm::UndefValue::get(llvmReturnType);

        const CollectionType *returnCollectionType = CastType<CollectionType>(returnType->GetBaseType());

        for (int i = 0; i < collectionType->GetElementCount(); ++i) {
            const PointerType *eltPtrType;
            llvm::Value *eltPtr = AddElementOffset(ptr, i, ptrType, "gather_elt_ptr", &eltPtrType);

            eltPtr = addVaryingOffsetsIfNeeded(eltPtr, eltPtrType);

            // It is a kludge. When we dereference varying pointer to uniform struct
            // with "bound uniform" member, we should return first unmasked member.
            int need_one_elem = CastType<StructType>(ptrType->GetBaseType()) &&
                                returnCollectionType->GetElementType(i)->IsUniformType();
            // This in turn will be another gather
            llvm::Value *eltValues = LoadInst(eltPtr, mask, eltPtrType, name, need_one_elem);

            retValue = InsertInst(retValue, eltValues, i, "set_value");
        }
        return retValue;
    } else if (ptrType->IsSlice()) {
        // If we have a slice pointer, we need to add the final slice
        // offset here right before issuing the actual gather
        //
        // FIXME: would it be better to do the corresponding same thing for
        // all of the varying offsets stuff here (and in scatter)?
        ptr = lFinalSliceOffset(this, ptr, &ptrType);
    }

    // Otherwise we should just have a basic scalar or pointer type and we
    // can go and do the actual gather
    AddInstrumentationPoint("gather");
    // Figure out which gather function to call based on the size of
    // the elements.
    const PointerType *pt = CastType<PointerType>(returnType);
    const char *funcName = NULL;
    if (pt != NULL)
        funcName = g->target->is32Bit() ? "__pseudo_gather32_i32" : "__pseudo_gather64_i64";
    // bool type is stored as i8.
    else if (returnType->IsBoolType())
        funcName = g->target->is32Bit() ? "__pseudo_gather32_i8" : "__pseudo_gather64_i8";
    else if (llvmReturnType == LLVMTypes::DoubleVectorType)
        funcName = g->target->is32Bit() ? "__pseudo_gather32_double" : "__pseudo_gather64_double";
    else if (llvmReturnType == LLVMTypes::Int64VectorType)
        funcName = g->target->is32Bit() ? "__pseudo_gather32_i64" : "__pseudo_gather64_i64";
    else if (llvmReturnType == LLVMTypes::FloatVectorType)
        funcName = g->target->is32Bit() ? "__pseudo_gather32_float" : "__pseudo_gather64_float";
    else if (llvmReturnType == LLVMTypes::Float16VectorType)
        funcName = g->target->is32Bit() ? "__pseudo_gather32_half" : "__pseudo_gather64_half";
    else if (llvmReturnType == LLVMTypes::Int32VectorType)
        funcName = g->target->is32Bit() ? "__pseudo_gather32_i32" : "__pseudo_gather64_i32";
    else if (llvmReturnType == LLVMTypes::Int16VectorType)
        funcName = g->target->is32Bit() ? "__pseudo_gather32_i16" : "__pseudo_gather64_i16";
    else {
        AssertPos(currentPos, llvmReturnType == LLVMTypes::Int8VectorType);
        funcName = g->target->is32Bit() ? "__pseudo_gather32_i8" : "__pseudo_gather64_i8";
    }

    llvm::Function *gatherFunc = m->module->getFunction(funcName);
    AssertPos(currentPos, gatherFunc != NULL);
#ifdef ISPC_XE_ENABLED
    if (emitXeHardwareMask()) {
        // Predicate ISPC mask with Xe execution mask so
        // after CMSimdCFLoweringPass pseudo_gather will have correct masked value.
        mask = XeSimdCFPredicate(mask);
    }
#endif

    llvm::Value *gatherCall = CallInst(gatherFunc, NULL, ptr, mask, name);

    // Add metadata about the source file location so that the
    // optimization passes can print useful performance warnings if we
    // can't optimize out this gather
    if (disableGSWarningCount == 0)
        addGSMetadata(gatherCall, currentPos);

    // bool type is stored as i8. So, it requires some processing.
    if (returnType->IsBoolType()) {
        if (g->target->getDataLayout()->getTypeSizeInBits(returnType->LLVMStorageType(g->ctx)) <
            g->target->getDataLayout()->getTypeSizeInBits(llvmReturnType)) {
            // This is needed when array of bool is passed in from cpp side
            // TRUE in clang is '1'. This is zero extended to i8.
            // In ispc, this is uniform * varying which after gather becomes
            // varying bool. Varying bool in ispc is '-1'. The most
            // significant bit being set to 1 is important for blendv
            // operations to work as expected.
            if (ptrType->GetBaseType()->IsUniformType()) {
                gatherCall = TruncInst(gatherCall, LLVMTypes::Int1VectorType);
                gatherCall = SExtInst(gatherCall, llvmReturnType);
            } else {
                gatherCall = SExtInst(gatherCall, llvmReturnType);
            }
        } else if (g->target->getDataLayout()->getTypeSizeInBits(returnType->LLVMStorageType(g->ctx)) >
                   g->target->getDataLayout()->getTypeSizeInBits(llvmReturnType)) {
            gatherCall = TruncInst(gatherCall, llvmReturnType);
        }
    }
    return gatherCall;
}

/** Add metadata to the given instruction to encode the current source file
    position.  This data is used in the lGetSourcePosFromMetadata()
    function in opt.cpp.
*/
void FunctionEmitContext::addGSMetadata(llvm::Value *v, SourcePos pos) {
    llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(v);
    if (inst == NULL)
        return;
    llvm::MDString *str = llvm::MDString::get(*g->ctx, pos.name);
    llvm::MDNode *md = llvm::MDNode::get(*g->ctx, str);
    inst->setMetadata("filename", md);

    llvm::Metadata *first_line = llvm::ConstantAsMetadata::get(LLVMInt32(pos.first_line));
    md = llvm::MDNode::get(*g->ctx, first_line);
    inst->setMetadata("first_line", md);

    llvm::Metadata *first_column = llvm::ConstantAsMetadata::get(LLVMInt32(pos.first_column));
    md = llvm::MDNode::get(*g->ctx, first_column);
    inst->setMetadata("first_column", md);

    llvm::Metadata *last_line = llvm::ConstantAsMetadata::get(LLVMInt32(pos.last_line));
    md = llvm::MDNode::get(*g->ctx, last_line);
    inst->setMetadata("last_line", md);

    llvm::Metadata *last_column = llvm::ConstantAsMetadata::get(LLVMInt32(pos.last_column));
    md = llvm::MDNode::get(*g->ctx, last_column);
    inst->setMetadata("last_column", md);
}

llvm::Value *FunctionEmitContext::AddrSpaceCast(llvm::Value *val, AddressSpace as, bool atEntryBlock) {
    Assert(llvm::isa<llvm::PointerType>(val->getType()));
    llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(val->getType());
    if (pt->getAddressSpace() == (unsigned)as) {
        return val;
    }
    llvm::PointerType *newType = llvm::PointerType::get(pt->getPointerElementType(), (unsigned)as);
    llvm::AddrSpaceCastInst *inst;
    if (atEntryBlock) {
        inst = new llvm::AddrSpaceCastInst(val, newType, val->getName() + "__cast", allocaBlock->getTerminator());
    } else {
        inst = new llvm::AddrSpaceCastInst(val, newType, val->getName() + "__cast", bblock);
    }

    return inst;
}

llvm::Value *FunctionEmitContext::AllocaInst(llvm::Type *llvmType, llvm::Value *size, const llvm::Twine &name,
                                             int align, bool atEntryBlock) {
    if ((llvmType == NULL) || (size == NULL)) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::AllocaInst *inst = NULL;
    unsigned AS = llvmFunction->getParent()->getDataLayout().getAllocaAddrSpace();
    if (atEntryBlock) {
        // We usually insert it right before the jump instruction at the
        // end of allocaBlock
        llvm::Instruction *retInst = allocaBlock->getTerminator();
        AssertPos(currentPos, retInst);
        inst = new llvm::AllocaInst(llvmType, AS, size, name, retInst);
    } else {
        // Unless the caller overrode the default and wants it in the
        // current basic block
        inst = new llvm::AllocaInst(llvmType, AS, size, name, bblock);
    }

    // If no alignment was specified but we have an array of a uniform
    // type, then align it to the native vector alignment; it's not
    // unlikely that this array will be loaded into varying variables with
    // what will be aligned accesses if the uniform -> varying load is done
    // in regular chunks.
    llvm::ArrayType *arrayType = llvm::dyn_cast<llvm::ArrayType>(llvmType);
    if (align == 0 && arrayType != NULL && !llvm::isa<llvm::VectorType>(arrayType->getElementType()))
        align = g->target->getNativeVectorAlignment();

    if (align != 0) {
        inst->setAlignment(llvm::MaybeAlign(align).valueOrOne());
    }
    return inst;
}

llvm::Value *FunctionEmitContext::AllocaInst(llvm::Type *llvmType, const llvm::Twine &name, int align,
                                             bool atEntryBlock) {
    if (llvmType == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::AllocaInst *inst = NULL;
    unsigned AS = llvmFunction->getParent()->getDataLayout().getAllocaAddrSpace();
    if (atEntryBlock) {
        // We usually insert it right before the jump instruction at the
        // end of allocaBlock
        llvm::Instruction *retInst = allocaBlock->getTerminator();
        AssertPos(currentPos, retInst);
        inst = new llvm::AllocaInst(llvmType, AS, name, retInst);
    } else {
        // Unless the caller overrode the default and wants it in the
        // current basic block
        inst = new llvm::AllocaInst(llvmType, AS, name, bblock);
    }

    // If no alignment was specified but we have an array of a uniform
    // type, then align it to the native vector alignment; it's not
    // unlikely that this array will be loaded into varying variables with
    // what will be aligned accesses if the uniform -> varying load is done
    // in regular chunks.
    llvm::ArrayType *arrayType = llvm::dyn_cast<llvm::ArrayType>(llvmType);
    if (align == 0 && arrayType != NULL && !llvm::isa<llvm::VectorType>(arrayType->getElementType()))
        align = g->target->getNativeVectorAlignment();

    if (align != 0) {
        inst->setAlignment(llvm::MaybeAlign(align).valueOrOne());
    }
    // Don't add debugging info to alloca instructions
    return inst;
}

llvm::Value *FunctionEmitContext::AllocaInst(const Type *ptrType, const llvm::Twine &name, int align,
                                             bool atEntryBlock) {
    if (ptrType == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::Type *llvmStorageType = ptrType->LLVMType(g->ctx);
    if ((((CastType<AtomicType>(ptrType) != NULL) || (CastType<VectorType>(ptrType) != NULL)) &&
         (ptrType->IsBoolType())) ||
        ((CastType<ArrayType>(ptrType) != NULL) && (ptrType->GetBaseType()->IsBoolType()))) {
        llvmStorageType = ptrType->LLVMStorageType(g->ctx);
    }

    return AllocaInst(llvmStorageType, name, align, atEntryBlock);
}

/** Code to store the given varying value to the given location, only
    storing the elements that correspond to active program instances as
    given by the provided storeMask value.  Note that the lvalue is only a
    single pointer, not a varying lvalue of one pointer per program
    instance (that case is handled by scatters).
 */
void FunctionEmitContext::maskedStore(llvm::Value *value, llvm::Value *ptr, const Type *ptrType, llvm::Value *mask) {
    if (value == NULL || ptr == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return;
    }

    AssertPos(currentPos, CastType<PointerType>(ptrType) != NULL);
    AssertPos(currentPos, ptrType->IsUniformType());

    const Type *valueType = ptrType->GetBaseType();
    const CollectionType *collectionType = CastType<CollectionType>(valueType);
    if (collectionType != NULL) {
        // Assigning a structure / array / vector. Handle each element
        // individually with what turns into a recursive call to
        // makedStore()
        for (int i = 0; i < collectionType->GetElementCount(); ++i) {
            const Type *eltType = collectionType->GetElementType(i);
            if (eltType == NULL) {
                Assert(m->errorCount > 0);
                continue;
            }
            llvm::Value *eltValue = ExtractInst(value, i, "value_member");
            llvm::Value *eltPtr = AddElementOffset(ptr, i, ptrType, "struct_ptr_ptr");
            const Type *eltPtrType = PointerType::GetUniform(eltType);
            StoreInst(eltValue, eltPtr, mask, eltType, eltPtrType);
        }
        return;
    }

    // We must have a regular atomic, enumerator, or pointer type at this
    // point.
    AssertPos(currentPos, Type::IsBasicType(valueType));
    valueType = valueType->GetAsNonConstType();

    // Figure out if we need a 8, 16, 32 or 64-bit masked store.
    llvm::Function *maskedStoreFunc = NULL;
    llvm::Type *llvmValueType = value->getType();
    llvm::Type *llvmValueStorageType = llvmValueType;

    const PointerType *pt = CastType<PointerType>(valueType);
    // bool type is stored as i8. So, it requires some processing.
    if ((pt == NULL) && (valueType->IsBoolType())) {
        llvmValueStorageType = LLVMTypes::BoolVectorStorageType;
    }
    if (pt != NULL) {
        if (pt->IsSlice()) {
            // Masked store of (varying) slice pointer.
            AssertPos(currentPos, pt->IsVaryingType());

            // First, extract the pointer from the slice struct and masked
            // store that.
            llvm::Value *v0 = ExtractInst(value, 0);
            llvm::Value *p0 = AddElementOffset(ptr, 0, ptrType);
            maskedStore(v0, p0, PointerType::GetUniform(pt->GetAsNonSlice()), mask);

            // And then do same for the integer offset
            llvm::Value *v1 = ExtractInst(value, 1);
            llvm::Value *p1 = AddElementOffset(ptr, 1, ptrType);
            const Type *offsetType = AtomicType::VaryingInt32;
            maskedStore(v1, p1, PointerType::GetUniform(offsetType), mask);

            return;
        }

        if (g->target->is32Bit())
            maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_i32");
        else
            maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_i64");
    } else if (llvmValueType == LLVMTypes::Int1VectorType) {
        llvm::Value *notMask = BinaryOperator(llvm::Instruction::Xor, mask, LLVMMaskAllOn, "~mask");
        llvm::Value *old = LoadInst(ptr, valueType);
        llvm::Value *maskedOld = BinaryOperator(llvm::Instruction::And, old, notMask, "old&~mask");
        llvm::Value *maskedNew = BinaryOperator(llvm::Instruction::And, value, mask, "new&mask");
        llvm::Value *final = BinaryOperator(llvm::Instruction::Or, maskedOld, maskedNew, "old_new_result");
        StoreInst(final, ptr, valueType);
        return;
    } else if (llvmValueStorageType == LLVMTypes::DoubleVectorType) {
        maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_double");
    } else if (llvmValueStorageType == LLVMTypes::Int64VectorType) {
        maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_i64");
    } else if (llvmValueStorageType == LLVMTypes::FloatVectorType) {
        maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_float");
    } else if (llvmValueStorageType == LLVMTypes::Float16VectorType) {
        maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_half");
    } else if (llvmValueStorageType == LLVMTypes::Int32VectorType) {
        maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_i32");
    } else if (llvmValueStorageType == LLVMTypes::Int16VectorType) {
        maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_i16");
    } else if (llvmValueStorageType == LLVMTypes::Int8VectorType) {
        maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_i8");
        value = SwitchBoolSize(value, llvmValueStorageType);
    }
    AssertPos(currentPos, maskedStoreFunc != NULL);

#ifdef ISPC_XE_ENABLED
    if (emitXeHardwareMask()) {
        mask = XeSimdCFPredicate(mask);
    }
#endif
    std::vector<llvm::Value *> args;
    args.push_back(ptr);
    args.push_back(value);
    args.push_back(mask);

    CallInst(maskedStoreFunc, NULL, args);
}

/** Scatter the given varying value to the locations given by the varying
    lvalue (which should be an array of pointers with size equal to the
    target's vector width.  We want to store each rvalue element at the
    corresponding pointer's location, *if* the mask for the corresponding
    program instance are on.  If they're off, don't do anything.
*/
void FunctionEmitContext::scatter(llvm::Value *value, llvm::Value *ptr, const Type *valueType, const Type *origPt,
                                  llvm::Value *mask) {
    const PointerType *ptrType = CastType<PointerType>(origPt);
    AssertPos(currentPos, ptrType != NULL);
    AssertPos(currentPos, ptrType->IsVaryingType());
    const CollectionType *srcCollectionType = CastType<CollectionType>(valueType);
    if (srcCollectionType != NULL) {
        // We're scattering a collection type--we need to keep track of the
        // source type (the type of the data values to be stored) and the
        // destination type (the type of objects in memory that will be
        // stored into) separately.  This is necessary so that we can get
        // all of the addressing calculations right if we're scattering
        // from a varying struct to an array of uniform instances of the
        // same struct type, versus scattering into an array of varying
        // instances of the struct type, etc.
        const CollectionType *dstCollectionType = CastType<CollectionType>(ptrType->GetBaseType());
        AssertPos(currentPos, dstCollectionType != NULL);

        // Scatter the collection elements individually
        for (int i = 0; i < srcCollectionType->GetElementCount(); ++i) {
            // First, get the values for the current element out of the
            // source.
            llvm::Value *eltValue = ExtractInst(value, i);
            const Type *srcEltType = srcCollectionType->GetElementType(i);

            // We may be scattering a uniform atomic element; in this case
            // we'll smear it out to be varying before making the recursive
            // scatter() call below.
            if (srcEltType->IsUniformType() && Type::IsBasicType(srcEltType)) {
                eltValue = SmearUniform(eltValue, "to_varying");
                srcEltType = srcEltType->GetAsVaryingType();
            }

            // Get the (varying) pointer to the i'th element of the target
            // collection
            llvm::Value *eltPtr = AddElementOffset(ptr, i, ptrType);

            // The destination element type may be uniform (e.g. if we're
            // scattering to an array of uniform structs).  Thus, we need
            // to be careful about passing the correct type to
            // addVaryingOffsetsIfNeeded() here.
            const Type *dstEltType = dstCollectionType->GetElementType(i);
            const PointerType *dstEltPtrType = PointerType::GetVarying(dstEltType);
            if (ptrType->IsSlice())
                dstEltPtrType = dstEltPtrType->GetAsSlice();

            eltPtr = addVaryingOffsetsIfNeeded(eltPtr, dstEltPtrType);

            // And recursively scatter() until we hit a basic type, at
            // which point the actual memory operations can be performed...
            scatter(eltValue, eltPtr, srcEltType, dstEltPtrType, mask);
        }
        return;
    } else if (ptrType->IsSlice()) {
        // As with gather, we need to add the final slice offset finally
        // once we get to a terminal SOA array of basic types..
        ptr = lFinalSliceOffset(this, ptr, &ptrType);
    }

    const PointerType *pt = CastType<PointerType>(valueType);

    // And everything should be a pointer or atomic (or enum) from here on out...
    AssertPos(currentPos,
              pt != NULL || CastType<AtomicType>(valueType) != NULL || CastType<EnumType>(valueType) != NULL);

    llvm::Type *llvmStorageType = value->getType();
    ;
    // bool type is stored as i8. So, it requires some processing.
    if ((pt == NULL) && (valueType->IsBoolType())) {
        llvmStorageType = LLVMTypes::BoolVectorStorageType;
        value = SwitchBoolSize(value, llvmStorageType);
    }
    const char *funcName = NULL;
    if (pt != NULL) {
        funcName = g->target->is32Bit() ? "__pseudo_scatter32_i32" : "__pseudo_scatter64_i64";
    } else if (llvmStorageType == LLVMTypes::DoubleVectorType) {
        funcName = g->target->is32Bit() ? "__pseudo_scatter32_double" : "__pseudo_scatter64_double";
    } else if (llvmStorageType == LLVMTypes::Int64VectorType) {
        funcName = g->target->is32Bit() ? "__pseudo_scatter32_i64" : "__pseudo_scatter64_i64";
    } else if (llvmStorageType == LLVMTypes::FloatVectorType) {
        funcName = g->target->is32Bit() ? "__pseudo_scatter32_float" : "__pseudo_scatter64_float";
    } else if (llvmStorageType == LLVMTypes::Float16VectorType) {
        funcName = g->target->is32Bit() ? "__pseudo_scatter32_half" : "__pseudo_scatter64_half";
    } else if (llvmStorageType == LLVMTypes::Int32VectorType) {
        funcName = g->target->is32Bit() ? "__pseudo_scatter32_i32" : "__pseudo_scatter64_i32";
    } else if (llvmStorageType == LLVMTypes::Int16VectorType) {
        funcName = g->target->is32Bit() ? "__pseudo_scatter32_i16" : "__pseudo_scatter64_i16";
    } else if (llvmStorageType == LLVMTypes::Int8VectorType) {
        funcName = g->target->is32Bit() ? "__pseudo_scatter32_i8" : "__pseudo_scatter64_i8";
    }

    llvm::Function *scatterFunc = m->module->getFunction(funcName);
    AssertPos(currentPos, scatterFunc != NULL);

    AddInstrumentationPoint("scatter");
#ifdef ISPC_XE_ENABLED
    if (emitXeHardwareMask()) {
        // Predicate ISPC mask with Xe execution mask so
        // after CMSimdCFLoweringPass pseudo_scatter will have correct masked value.
        mask = XeSimdCFPredicate(mask);
    }
#endif
    std::vector<llvm::Value *> args;
    args.push_back(ptr);
    args.push_back(value);
    args.push_back(mask);
    llvm::Value *inst = CallInst(scatterFunc, NULL, args);

    if (disableGSWarningCount == 0)
        addGSMetadata(inst, currentPos);
}

void FunctionEmitContext::StoreInst(llvm::Value *value, llvm::Value *ptr, const Type *ptrType, bool isUniformData) {
    if (value == NULL || ptr == NULL) {
        // may happen due to error elsewhere
        AssertPos(currentPos, m->errorCount > 0);
        return;
    }

    llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(ptr->getType());
    AssertPos(currentPos, pt != NULL);

    if ((ptrType != NULL) && (ptrType->IsBoolType())) {
        if ((CastType<AtomicType>(ptrType) != NULL)) {
            value = SwitchBoolSize(value, ptrType->LLVMStorageType(g->ctx));
        } else if (CastType<VectorType>(ptrType) != NULL) {
            const VectorType *vType = CastType<VectorType>(ptrType);
            if (CastType<AtomicType>(vType->GetElementType()) != NULL) {
                value = SwitchBoolSize(value, ptrType->LLVMStorageType(g->ctx));
            }
        }
    }

    llvm::StoreInst *inst = new llvm::StoreInst(value, ptr, bblock);

    if (g->opt.forceAlignedMemory && llvm::dyn_cast<llvm::VectorType>(pt->getElementType())) {
        inst->setAlignment(llvm::MaybeAlign(g->target->getNativeVectorAlignment()).valueOrOne());
    }

#ifdef ISPC_XE_ENABLED
    // For uniform data like short vectors we need to add ISPC-Uniform metadata
    // to exclude these instructions from predication in SIMDCFLowering pass.
    llvm::VectorType *ty = llvm::dyn_cast<llvm::VectorType>(value->getType());
    if (ty != NULL) {
        if (emitXeHardwareMask() && isUniformData) {
            XeUniformMetadata(inst);
        }
    }
#endif
    AddDebugPos(inst);
}

void FunctionEmitContext::StoreInst(llvm::Value *value, llvm::Value *ptr, llvm::Value *mask, const Type *valueType,
                                    const Type *ptrRefType) {
    if (value == NULL || ptr == NULL) {
        // may happen due to error elsewhere
        AssertPos(currentPos, m->errorCount > 0);
        return;
    }

    const PointerType *ptrType;
    if (CastType<ReferenceType>(ptrRefType) != NULL)
        ptrType = PointerType::GetUniform(ptrRefType->GetReferenceTarget());
    else {
        ptrType = CastType<PointerType>(ptrRefType);
        AssertPos(currentPos, ptrType != NULL);
    }

    if (CastType<UndefinedStructType>(ptrType->GetBaseType())) {
        Error(currentPos, "Unable to store to undefined struct type \"%s\".",
              ptrType->GetBaseType()->GetString().c_str());
        return;
    }

    // Figure out what kind of store we're doing here
    if (ptrType->IsUniformType()) {
        if (ptrType->IsSlice())
            // storing a uniform value to a single slice of a SOA type
            storeUniformToSOA(value, ptr, mask, valueType, ptrType);
        else if (ptrType->GetBaseType()->IsUniformType())
            // the easy case
            StoreInst(value, ptr, valueType, true);
        else if (mask == LLVMMaskAllOn && !g->opt.disableMaskAllOnOptimizations)
            // Otherwise it is a masked store unless we can determine that the
            // mask is all on...  (Unclear if this check is actually useful.)
            StoreInst(value, ptr, valueType, false);
        else {
            maskedStore(value, ptr, ptrType, mask);
        }
    } else {
        AssertPos(currentPos, ptrType->IsVaryingType());
        // We have a varying ptr (an array of pointers), so it's time to
        // scatter
        scatter(value, ptr, valueType, ptrType, GetFullMask());
    }
}

/** Store a uniform type to SOA-laid-out memory.
 */
void FunctionEmitContext::storeUniformToSOA(llvm::Value *value, llvm::Value *ptr, llvm::Value *mask,
                                            const Type *valueType, const PointerType *ptrType) {
    AssertPos(currentPos, Type::EqualIgnoringConst(ptrType->GetBaseType()->GetAsUniformType(), valueType));

    const CollectionType *ct = CastType<CollectionType>(valueType);
    if (ct != NULL) {
        // Handle collections element wise...
        for (int i = 0; i < ct->GetElementCount(); ++i) {
            llvm::Value *eltValue = ExtractInst(value, i);
            const Type *eltType = ct->GetElementType(i);
            const PointerType *dstEltPtrType;
            llvm::Value *dstEltPtr = AddElementOffset(ptr, i, ptrType, "slice_offset", &dstEltPtrType);
            StoreInst(eltValue, dstEltPtr, mask, eltType, dstEltPtrType);
        }
    } else {
        // We're finally at a leaf SOA array; apply the slice offset and
        // then we can do a final regular store
        AssertPos(currentPos, Type::IsBasicType(valueType));
        ptr = lFinalSliceOffset(this, ptr, &ptrType);
        StoreInst(value, ptr, valueType, valueType->IsUniformType());
    }
}

void FunctionEmitContext::MemcpyInst(llvm::Value *dest, llvm::Value *src, llvm::Value *count, llvm::Value *align) {
    dest = BitCastInst(dest, LLVMTypes::VoidPointerType);
    src = BitCastInst(src, LLVMTypes::VoidPointerType);
    if (count->getType() != LLVMTypes::Int64Type) {
        AssertPos(currentPos, count->getType() == LLVMTypes::Int32Type);
        count = ZExtInst(count, LLVMTypes::Int64Type, "count_to_64");
    }
    if (align == NULL)
        align = LLVMInt32(1);
    llvm::FunctionCallee mcFuncCallee =
        m->module->getOrInsertFunction("llvm.memcpy.p0i8.p0i8.i64", LLVMTypes::VoidType, LLVMTypes::VoidPointerType,
                                       LLVMTypes::VoidPointerType, LLVMTypes::Int64Type, LLVMTypes::BoolType);
    llvm::Constant *mcFunc = llvm::cast<llvm::Constant>(mcFuncCallee.getCallee());
    AssertPos(currentPos, mcFunc != NULL);
    AssertPos(currentPos, llvm::isa<llvm::Function>(mcFunc));

    std::vector<llvm::Value *> args;
    args.push_back(dest);
    args.push_back(src);
    args.push_back(count);
    args.push_back(LLVMFalse); /* not volatile */
#ifdef ISPC_XE_ENABLED
    llvm::Value *callinst = CallInst(mcFunc, NULL, args, "");
    if (emitXeHardwareMask()) {
        XeUniformMetadata(callinst);
    }
#else
    CallInst(mcFunc, NULL, args, "");
#endif
}

void FunctionEmitContext::setLoopUnrollMetadata(llvm::Instruction *inst,
                                                std::pair<Globals::pragmaUnrollType, int> loopAttribute,
                                                SourcePos pos) {
    if (inst == NULL) {
        return;
    }

    if (loopAttribute.first == Globals::pragmaUnrollType::none) {
        return;
    }

    llvm::SmallVector<llvm::Metadata *, 4> Args;
    llvm::TempMDTuple TempNode = llvm::MDNode::getTemporary(*g->ctx, llvm::None);
    Args.push_back(TempNode.get());
    if (loopAttribute.first == Globals::pragmaUnrollType::count) {
        llvm::Metadata *Vals[] = {llvm::MDString::get(*g->ctx, "llvm.loop.unroll.count"),
                                  llvm::ConstantAsMetadata::get(LLVMInt32(loopAttribute.second))};
        Args.push_back(llvm::MDNode::get(*g->ctx, Vals));
    } else if (loopAttribute.first == Globals::pragmaUnrollType::unroll) {
        llvm::Metadata *Vals[] = {llvm::MDString::get(*g->ctx, "llvm.loop.unroll.enable")};
        Args.push_back(llvm::MDNode::get(*g->ctx, Vals));
    } else if (loopAttribute.first == Globals::pragmaUnrollType::nounroll) {
        llvm::Metadata *Vals[] = {llvm::MDString::get(*g->ctx, "llvm.loop.unroll.disable")};
        Args.push_back(llvm::MDNode::get(*g->ctx, Vals));
    }
    llvm::MDNode *LoopID = llvm::MDNode::getDistinct(*g->ctx, Args);
    LoopID->replaceOperandWith(0, LoopID);
    inst->setMetadata("llvm.loop", LoopID);
}

llvm::Instruction *FunctionEmitContext::BranchInst(llvm::BasicBlock *dest) {
    llvm::Instruction *b = llvm::BranchInst::Create(dest, bblock);
    AddDebugPos(b);
    return b;
}

llvm::Instruction *FunctionEmitContext::BranchInst(llvm::BasicBlock *trueBlock, llvm::BasicBlock *falseBlock,
                                                   llvm::Value *test) {
    llvm::Instruction *b = NULL;
    if (test == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return b;
    }
#ifdef ISPC_XE_ENABLED
    if (emitXeHardwareMask())
        test = XePrepareVectorBranch(test);
#endif
    b = llvm::BranchInst::Create(trueBlock, falseBlock, test, bblock);
    AddDebugPos(b);
    return b;
}

llvm::Value *FunctionEmitContext::ExtractInst(llvm::Value *v, int elt, const llvm::Twine &name) {
    if (v == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::Instruction *ei = NULL;
    if (llvm::isa<llvm::VectorType>(v->getType()))
        ei = llvm::ExtractElementInst::Create(
            v, LLVMInt32(elt),
            name.isTriviallyEmpty() ? ((llvm::Twine(v->getName()) + "_extract_") + llvm::Twine(elt)) : name, bblock);
    else
        ei = llvm::ExtractValueInst::Create(
            v, elt, name.isTriviallyEmpty() ? ((llvm::Twine(v->getName()) + "_extract_") + llvm::Twine(elt)) : name,
            bblock);
    AddDebugPos(ei);
    return ei;
}

llvm::Value *FunctionEmitContext::InsertInst(llvm::Value *v, llvm::Value *eltVal, int elt, const llvm::Twine &name) {
    if (v == NULL || eltVal == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::Instruction *ii = NULL;
    if (llvm::isa<llvm::VectorType>(v->getType()))
        ii = llvm::InsertElementInst::Create(
            v, eltVal, LLVMInt32(elt),
            name.isTriviallyEmpty() ? ((llvm::Twine(v->getName()) + "_insert_") + llvm::Twine(elt)) : name, bblock);
    else
        ii = llvm::InsertValueInst::Create(
            v, eltVal, elt,
            name.isTriviallyEmpty() ? ((llvm::Twine(v->getName()) + "_insert_") + llvm::Twine(elt)) : name, bblock);
    AddDebugPos(ii);
    return ii;
}

llvm::Value *FunctionEmitContext::ShuffleInst(llvm::Value *v1, llvm::Value *v2, llvm::Value *mask,
                                              const llvm::Twine &name) {
    if (v1 == NULL || v2 == NULL || mask == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::Instruction *ii = new llvm::ShuffleVectorInst(
        v1, v2, mask, name.isTriviallyEmpty() ? (llvm::Twine(v1->getName()) + "_shuffle") : name, bblock);

    AddDebugPos(ii);
    return ii;
}

llvm::Value *FunctionEmitContext::BroadcastValue(llvm::Value *v, llvm::Type *vecType, const llvm::Twine &name) {
    if (v == NULL || vecType == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    llvm::FixedVectorType *ty = llvm::dyn_cast<llvm::FixedVectorType>(vecType);
    Assert(ty && ty->getElementType() == v->getType());
#else
    llvm::VectorType *ty = llvm::dyn_cast<llvm::VectorType>(vecType);
    Assert(ty && ty->getVectorElementType() == v->getType());
#endif

    // Generate the following sequence:
    //   %name_init.i = insertelement <4 x i32> undef, i32 %val, i32 0
    //   %name.i = shufflevector <4 x i32> %name_init.i, <4 x i32> undef,
    //                                              <4 x i32> zeroinitializer

    llvm::Value *undef1 = llvm::UndefValue::get(vecType);
    llvm::Value *undef2 = llvm::UndefValue::get(vecType);

    // InsertElement
    llvm::Value *insert =
        InsertInst(undef1, v, 0, name.isTriviallyEmpty() ? (llvm::Twine(v->getName()) + "_broadcast") : name + "_init");

    // ShuffleVector
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
    llvm::Constant *zeroVec = llvm::ConstantVector::getSplat(
        vecType->getVectorNumElements(), llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*g->ctx)));
#elif ISPC_LLVM_VERSION < ISPC_LLVM_12_0
    llvm::Constant *zeroVec =
        llvm::ConstantVector::getSplat({static_cast<unsigned int>(ty->getNumElements()), false},
                                       llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*g->ctx)));
#else
    llvm::Constant *zeroVec =
        llvm::ConstantVector::getSplat(llvm::ElementCount::get(static_cast<unsigned int>(ty->getNumElements()), false),
                                       llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*g->ctx)));
#endif
    llvm::Value *ret = ShuffleInst(insert, undef2, zeroVec,
                                   name.isTriviallyEmpty() ? (llvm::Twine(v->getName()) + "_broadcast") : name);

    return ret;
}

llvm::PHINode *FunctionEmitContext::PhiNode(llvm::Type *type, int count, const llvm::Twine &name) {
    llvm::PHINode *pn = llvm::PHINode::Create(type, count, name.isTriviallyEmpty() ? "phi" : name, bblock);
    AddDebugPos(pn);
    return pn;
}

llvm::Instruction *FunctionEmitContext::SelectInst(llvm::Value *test, llvm::Value *val0, llvm::Value *val1,
                                                   const llvm::Twine &name) {
    if (test == NULL || val0 == NULL || val1 == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    llvm::Instruction *inst = llvm::SelectInst::Create(
        test, val0, val1, name.isTriviallyEmpty() ? (llvm::Twine(test->getName()) + "_select") : name, bblock);
    AddDebugPos(inst);
    return inst;
}

/** Given a value representing a function to be called or possibly-varying
    pointer to a function to be called, figure out how many arguments the
    function has. */
static unsigned int lCalleeArgCount(llvm::Value *callee, const FunctionType *funcType) {
    llvm::FunctionType *ft = llvm::dyn_cast<llvm::FunctionType>(callee->getType());

    if (ft == NULL) {
        llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(callee->getType());
        if (pt == NULL) {
            // varying--in this case, it must be the version of the
            // function that takes a mask
            return funcType->GetNumParameters() + 1;
        }
        ft = llvm::dyn_cast<llvm::FunctionType>(pt->getElementType());
    }

    Assert(ft != NULL);
    return ft->getNumParams();
}

llvm::Value *FunctionEmitContext::CallInst(llvm::Value *func, const FunctionType *funcType,
                                           const std::vector<llvm::Value *> &args, const llvm::Twine &name) {
    if (func == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }
    std::vector<llvm::Value *> argVals;
    // Most of the time, the mask is passed as the last argument.  this
    // isn't the case for things like intrinsics, builtins, and extern "C"
    // functions from the application.  Add the mask if it's needed.
    // There may be more arguments than function parameters for vararg case.
    unsigned int calleeArgCount = lCalleeArgCount(func, funcType);

    // If we have ISPC external function without mask, we should cast all
    // pointers to generic address space before call.
    llvm::Function *f = llvm::dyn_cast<llvm::Function>(func);
    if (funcType && funcType->RequiresAddrSpaceCasts(f)) {
        for (llvm::Value *arg : args) {
            if (llvm::isa<llvm::PointerType>(arg->getType())) {
                llvm::Value *adrCast = AddrSpaceCast(arg, AddressSpace::ispc_generic);
                argVals.push_back(adrCast);
            } else {
                argVals.push_back(arg);
            }
        }
    } else {
        argVals = args;
    }

    AssertPos(currentPos, (llvm::isa<llvm::Function>(func) && llvm::cast<llvm::Function>(func)->isVarArg()) ||
                              argVals.size() + 1 == calleeArgCount || argVals.size() == calleeArgCount);
    if (argVals.size() + 1 == calleeArgCount) {
        llvm::Value *mask = NULL;

#ifdef ISPC_XE_ENABLED
        if (emitXeHardwareMask())
            // This will create mask according to current EM on SIMD CF Lowering.
            // The result will be like       mask = select (EM, AllOn, AllFalse)
            mask = XeSimdCFPredicate(LLVMMaskAllOn);
        else
#endif
            mask = GetFullMask();
        argVals.push_back(mask);
    }

    if (llvm::isa<llvm::VectorType>(func->getType()) == false) {
        // Regular 'uniform' function call--just one function or function
        // pointer, so just emit the IR directly.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
        llvm::PointerType *func_ptr_type = llvm::dyn_cast<llvm::PointerType>(func->getType());
        llvm::FunctionType *func_type = llvm::dyn_cast<llvm::FunctionType>(func_ptr_type->getPointerElementType());
        llvm::CallInst *callinst = llvm::CallInst::Create(func_type, func, argVals, name, bblock);
#else
        llvm::CallInst *callinst = llvm::CallInst::Create(func, argVals, name, bblock);
#endif

        // We could be dealing with a function pointer in which case this will not be a 'llvm::Function'.
        // If 'llvm::Function', use same calling convention as the actual function definition. It's
        // important we do this since  prebuilt stdlib functions does not use vectorcall on any OS.
        // If function pointer, it's safe to assume  that we use the cached calling convention
        // since this has to be a user defined function.
        llvm::Function *funcForConv = llvm::dyn_cast<llvm::Function>(func);

        if (g->calling_conv == CallingConv::x86_vectorcall) {
            if (funcForConv) {
                callinst->setCallingConv(funcForConv->getCallingConv());
            } else {
                callinst->setCallingConv(llvm::CallingConv::X86_VectorCall);
            }
        }

        llvm::Instruction *ci = callinst;

        // Copy noalias attribute to call instruction, to enable better
        // alias analysis.
        // TODO: what other attributes needs to be copied?
        // TODO: do the same for varing path.
        llvm::CallInst *cc = llvm::dyn_cast<llvm::CallInst>(ci);
        if (cc && cc->getCalledFunction()) {
            if (cc->getCalledFunction()->returnDoesNotAlias()) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
                cc->addRetAttr(llvm::Attribute::NoAlias);
#else
                cc->addAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias);
#endif
            }
            // TO DO:Add x86 changes as a separate commit
            /* unsigned int argSize = cc->arg_size();
            llvm::Function *calledFunc = cc->getCalledFunction();
            for (int argNum = 0; argNum < argSize; argNum++) {
                if (calledFunc->getArg(argNum)->hasAttribute(llvm::Attribute::InReg))
                    cc->addParamAttr(argNum, llvm::Attribute::InReg);
            }*/
        }

        AddDebugPos(ci);
        return ci;
    } else {
        // Emit the code for a varying function call, where we have an
        // vector of function pointers, one for each program instance.  The
        // basic strategy is that we go through the function pointers, and
        // for the executing program instances, for each unique function
        // pointer that's in the vector, call that function with a mask
        // equal to the set of active program instances that also have that
        // function pointer.  When all unique function pointers have been
        // called, we're done.
        llvm::BasicBlock *bbTest = CreateBasicBlock("varying_funcall_test", GetCurrentBasicBlock());
        llvm::BasicBlock *bbCall = CreateBasicBlock("varying_funcall_call", bbTest);
        llvm::BasicBlock *bbDone = CreateBasicBlock("varying_funcall_done", bbCall);

        llvm::BasicBlock *bbSIMDCall = NULL;
        llvm::BasicBlock *bbSIMDCallJoin = NULL;
        if (emitXeHardwareMask()) {
            bbSIMDCall = CreateBasicBlock("varying_funcall_simd_call", bbCall);
            bbSIMDCallJoin = CreateBasicBlock("varying_funcall_simd_call_join", bbSIMDCall);
        }

        // Get the current mask value so we can restore it later
        llvm::Value *origMask = GetInternalMask();

        // First allocate memory to accumulate the various program
        // instances' return values...
        Assert(funcType != NULL);
        const Type *returnType = funcType->GetReturnType();
        llvm::Type *llvmReturnType = returnType->LLVMType(g->ctx);
        llvm::Value *resultPtr = NULL;
        if (llvmReturnType->isVoidTy() == false)
            resultPtr = AllocaInst(returnType);

        // The memory pointed to by maskPointer tracks the set of program
        // instances for which we still need to call the function they are
        // pointing to.  It starts out initialized with the mask of
        // currently running program instances.
        llvm::Value *oldFullMask = NULL;
        llvm::Value *maskPtr = AllocaInst(LLVMTypes::MaskType);
        if (emitXeHardwareMask()) {
#ifdef ISPC_XE_ENABLED
            // Current mask will be calculated according to EM mask
            oldFullMask = XeSimdCFPredicate(LLVMMaskAllOn);
            StoreInst(oldFullMask, maskPtr, NULL, true);
#endif
        } else {
            oldFullMask = GetFullMask();
            StoreInst(oldFullMask, maskPtr);
        }

        // Mask wasn't initialized
        Assert(oldFullMask != NULL && "Mask is not initialized");

        // And now we branch to the test to see if there's more work to be
        // done.
        BranchInst(bbTest);

        // bbTest: are any lanes of the mask still on?  If so, jump to
        // bbCall
        SetCurrentBasicBlock(bbTest);
        {
            llvm::Value *maskLoad = LoadInst(maskPtr);
            llvm::Value *any = Any(maskLoad);
            BranchInst(bbCall, bbDone, any);
        }

        // bbCall: this is the body of the loop that calls out to one of
        // the active function pointer values.
        SetCurrentBasicBlock(bbCall);
        {
            // Figure out the first lane that still needs its function
            // pointer to be called.
            llvm::Value *currentMask = LoadInst(maskPtr);
            llvm::Function *cttz = m->module->getFunction("__count_trailing_zeros_i64");
            AssertPos(currentPos, cttz != NULL);
            llvm::Value *firstLane64 = CallInst(cttz, NULL, LaneMask(currentMask), "first_lane64");
            llvm::Value *firstLane = TruncInst(firstLane64, LLVMTypes::Int32Type, "first_lane32");

            // Get the pointer to the function we're going to call this
            // time through: ftpr = func[firstLane]
            llvm::Value *fptr = llvm::ExtractElementInst::Create(func, firstLane, "extract_fptr", bblock);

            // Smear it out into an array of function pointers
            llvm::Value *fptrSmear = SmearUniform(fptr, "func_ptr");

            // fpOverlap = (fpSmearAsVec == fpOrigAsVec).  This gives us a
            // mask for the set of program instances that have the same
            // value for their function pointer.
            llvm::Value *fpOverlap = CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, fptrSmear, func);
            fpOverlap = I1VecToBoolVec(fpOverlap);

            // Figure out the mask to use when calling the function
            // pointer: we need to AND the current execution mask to handle
            // the case of any non-running program instances that happen to
            // have this function pointer value.
            // callMask = (currentMask & fpOverlap)
            llvm::Value *callMask = BinaryOperator(llvm::Instruction::And, currentMask, fpOverlap, "call_mask");

            if (emitXeHardwareMask()) {
                // TODO: Seems like it is possible to move code
                // from bbSIMDCallJoin block here. Decide if
                // it should be done.

                // Execution is performed according to EM for Xe.
                // Branch to BB where EM is applied for call.
                BranchInst(bbSIMDCall, bbSIMDCallJoin, callMask);
                // Emit code for call
                SetCurrentBasicBlock(bbSIMDCall);
            } else {
                // Set the mask
                SetInternalMask(callMask);
            }

            // bitcast the i32/64 function pointer to the actual function
            // pointer type.
            llvm::Type *llvmFuncType = funcType->LLVMFunctionType(g->ctx);
            llvm::Type *llvmFPtrType = llvm::PointerType::get(llvmFuncType, 0);
            llvm::Value *fptrCast = IntToPtrInst(fptr, llvmFPtrType);

            // Call the function: callResult = call ftpr(args, args, call mask)
            llvm::Value *callResult = CallInst(fptrCast, funcType, args, name);

            // Now, do a masked store into the memory allocated to
            // accumulate the result using the call mask.
            if (callResult != NULL && callResult->getType() != LLVMTypes::VoidType) {
                AssertPos(currentPos, resultPtr != NULL);
                if (emitXeHardwareMask()) {
                    // This store will be predicated during SIMD CF Lowering
                    StoreInst(callResult, resultPtr);
                } else {
                    StoreInst(callResult, resultPtr, callMask, returnType, PointerType::GetUniform(returnType));
                }
            } else
                AssertPos(currentPos, resultPtr == NULL);

            if (emitXeHardwareMask()) {
                // Finish SIMDCall BB
                BranchInst(bbSIMDCallJoin);
                SetCurrentBasicBlock(bbSIMDCallJoin);
            }

            // Update the mask to turn off the program instances for which
            // we just called the function.
            // currentMask = currentMask & ~callmask
            llvm::Value *notCallMask = BinaryOperator(llvm::Instruction::Xor, callMask, LLVMMaskAllOn, "~callMask");
            currentMask = BinaryOperator(llvm::Instruction::And, currentMask, notCallMask, "currentMask&~callMask");
            if (emitXeHardwareMask()) {
                StoreInst(currentMask, maskPtr, NULL, true);
            } else {
                StoreInst(currentMask, maskPtr);
            }

            // And go back to the test to see if we need to do another
            // call.
            BranchInst(bbTest);
        }

        // bbDone: We're all done; clean up and return the result we've
        // accumulated in the result memory.
        SetCurrentBasicBlock(bbDone);
        SetInternalMask(origMask);
        return resultPtr ? LoadInst(resultPtr, funcType->GetReturnType()) : NULL;
    }
}

llvm::Value *FunctionEmitContext::CallInst(llvm::Value *func, const FunctionType *funcType, llvm::Value *arg,
                                           const llvm::Twine &name) {
    std::vector<llvm::Value *> args;
    args.push_back(arg);
    return CallInst(func, funcType, args, name);
}

llvm::Value *FunctionEmitContext::CallInst(llvm::Value *func, const FunctionType *funcType, llvm::Value *arg0,
                                           llvm::Value *arg1, const llvm::Twine &name) {
    std::vector<llvm::Value *> args;
    args.push_back(arg0);
    args.push_back(arg1);
    return CallInst(func, funcType, args, name);
}

llvm::Instruction *FunctionEmitContext::ReturnInst() {
    if (launchedTasks)
        // Add a sync call at the end of any function that launched tasks
        SyncInst();

#ifdef ISPC_XE_ENABLED
    if (emitXeHardwareMask()) {
        // Branch to return point. It will turn off lanes
        // in varying CF. For uniform CF it will be considered
        // as usual jmp.
        // TODO: this is a temporary workaround and will be
        // changed with SPIR-V emitting solution
        BranchInst(returnPoint);
        bblock = NULL;
        // We don't actually create return instruction here
        return NULL;
    }
#endif

    llvm::Instruction *rinst = NULL;
    if (returnValuePtr != NULL) {
        // We have value(s) to return; load them from their storage
        // location
        llvm::Value *retVal = LoadInst(returnValuePtr, function->GetReturnType(), "return_value");
        rinst = llvm::ReturnInst::Create(*g->ctx, retVal, bblock);
    } else {
        AssertPos(currentPos, function->GetReturnType()->IsVoidType());
        rinst = llvm::ReturnInst::Create(*g->ctx, bblock);
    }

    AddDebugPos(rinst);
    bblock = NULL;
    return rinst;
}

llvm::Value *FunctionEmitContext::LaunchInst(llvm::Value *callee, std::vector<llvm::Value *> &argVals,
                                             llvm::Value *launchCount[3], const FunctionType *funcType) {
    if (g->target->isXeTarget()) {
        Error(currentPos, "\"launch\" keyword is not supported for Xe targets");
        return NULL;
    }

    if (callee == NULL) {
        AssertPos(currentPos, m->errorCount > 0);
        return NULL;
    }

    if (!(llvm::isa<llvm::Function>(callee) || llvm::isa<llvm::PointerType>(callee->getType()))) {
        Error(currentPos, "Must provide function name or uniform function pointer to \"task\"-qualified function for "
                          "\"launch\" expression");
        return NULL;
    }

    launchedTasks = true;

    AssertPos(currentPos, funcType != NULL);
    llvm::Type *llvmFuncType = funcType->LLVMFunctionType(g->ctx);
    AssertPos(currentPos, funcType->LLVMFunctionType(g->ctx)->getFunctionNumParams() > 0);
    llvm::Type *argType = llvmFuncType->getFunctionParamType(0);
    AssertPos(currentPos, llvm::PointerType::classof(argType));

    llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(argType);
    AssertPos(currentPos, pt);
    AssertPos(currentPos, llvm::StructType::classof(pt->getElementType()));
    llvm::StructType *argStructType = static_cast<llvm::StructType *>(pt->getElementType());

    llvm::Function *falloc = m->module->getFunction("ISPCAlloc");
    AssertPos(currentPos, falloc != NULL);
    llvm::Value *structSize = g->target->SizeOf(argStructType, bblock);
    if (structSize->getType() != LLVMTypes::Int64Type)
        // ISPCAlloc expects the size as an uint64_t, but on 32-bit
        // targets, SizeOf returns a 32-bit value
        structSize = ZExtInst(structSize, LLVMTypes::Int64Type, "struct_size_to_64");
    int align = 4 * RoundUpPow2(g->target->getNativeVectorWidth());

    std::vector<llvm::Value *> allocArgs;
    allocArgs.push_back(launchGroupHandlePtr);
    allocArgs.push_back(structSize);
    allocArgs.push_back(LLVMInt32(align));
    llvm::Value *voidmem = CallInst(falloc, NULL, allocArgs, "args_ptr");
    llvm::Value *argmem = BitCastInst(voidmem, pt);

    // Copy the values of the parameters into the appropriate place in
    // the argument block
    for (unsigned int i = 0; i < argVals.size(); ++i) {
        llvm::Value *ptr = AddElementOffset(argmem, i, NULL, "funarg");
        // don't need to do masked store here, I think
        StoreInst(argVals[i], ptr);
    }

    if (argStructType->getNumElements() == argVals.size() + 1) {
        // copy in the mask
        llvm::Value *mask = GetFullMask();
        llvm::Value *ptr = AddElementOffset(argmem, argVals.size(), NULL, "funarg_mask");
        StoreInst(mask, ptr);
    }

    // And emit the call to the user-supplied task launch function, passing
    // a pointer to the task function being called and a pointer to the
    // argument block we just filled in
    llvm::Value *fptr = BitCastInst(callee, LLVMTypes::VoidPointerType);
    llvm::Function *flaunch = m->module->getFunction("ISPCLaunch");
    AssertPos(currentPos, flaunch != NULL);
    std::vector<llvm::Value *> args;
    args.push_back(launchGroupHandlePtr);
    args.push_back(fptr);
    args.push_back(voidmem);
    args.push_back(launchCount[0]);
    args.push_back(launchCount[1]);
    args.push_back(launchCount[2]);
    return CallInst(flaunch, NULL, args, "");
}

void FunctionEmitContext::SyncInst() {
    if (g->target->isXeTarget()) {
        Error(currentPos, "\"sync\" keyword is not supported for Xe targets");
        return;
    }

    llvm::Value *launchGroupHandle = LoadInst(launchGroupHandlePtr);
    llvm::Value *nullPtrValue = llvm::Constant::getNullValue(LLVMTypes::VoidPointerType);
    llvm::Value *nonNull = CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE, launchGroupHandle, nullPtrValue);
    llvm::BasicBlock *bSync = CreateBasicBlock("call_sync");
    llvm::BasicBlock *bPostSync = CreateBasicBlock("post_sync");
    BranchInst(bSync, bPostSync, nonNull);

    SetCurrentBasicBlock(bSync);
    llvm::Function *fsync = m->module->getFunction("ISPCSync");
    if (fsync == NULL)
        FATAL("Couldn't find ISPCSync declaration?!");
    CallInst(fsync, NULL, launchGroupHandle, "");

    // zero out the handle so that if ISPCLaunch is called again in this
    // function, it knows it's starting out from scratch
    StoreInst(nullPtrValue, launchGroupHandlePtr);

    BranchInst(bPostSync);

    SetCurrentBasicBlock(bPostSync);
}

/** When we gathering from or scattering to a varying atomic type, we need
    to add an appropriate offset to the final address for each lane right
    before we use it.  Given a varying pointer we're about to use and its
    type, this function determines whether these offsets are needed and
    returns an updated pointer that incorporates these offsets if needed.
 */
llvm::Value *FunctionEmitContext::addVaryingOffsetsIfNeeded(llvm::Value *ptr, const Type *ptrType) {
    // This should only be called for varying pointers
    const PointerType *pt = CastType<PointerType>(ptrType);
    AssertPos(currentPos, pt && pt->IsVaryingType());

    const Type *baseType = ptrType->GetBaseType();
    if (Type::IsBasicType(baseType) == false)
        return ptr;

    if (baseType->IsVaryingType() == false)
        return ptr;

    // Find the size of a uniform element of the varying type
    llvm::Type *llvmBaseUniformType = baseType->GetAsUniformType()->LLVMType(g->ctx);
    llvm::Value *unifSize = g->target->SizeOf(llvmBaseUniformType, bblock);
    unifSize = SmearUniform(unifSize);

    // Compute offset = <0, 1, .. > * unifSize
    bool is32bits = g->target->is32Bit() || g->opt.force32BitAddressing;
    llvm::Value *varyingOffsets = ProgramIndexVector(is32bits);

    llvm::Value *offset = BinaryOperator(llvm::Instruction::Mul, unifSize, varyingOffsets);

    if (g->opt.force32BitAddressing == true && g->target->is32Bit() == false)
        // On 64-bit targets where we're doing 32-bit addressing
        // calculations, we need to convert to an i64 vector before adding
        // to the pointer
        offset = SExtInst(offset, LLVMTypes::Int64VectorType, "offset_to_64");

    return BinaryOperator(llvm::Instruction::Add, ptr, offset);
}

CFInfo *FunctionEmitContext::popCFState() {
    AssertPos(currentPos, controlFlowInfo.size() > 0);
    CFInfo *ci = controlFlowInfo.back();
    controlFlowInfo.pop_back();

    if (ci->IsSwitch()) {
        breakTarget = ci->savedBreakTarget;
        continueTarget = ci->savedContinueTarget;
        breakLanesPtr = ci->savedBreakLanesPtr;
        continueLanesPtr = ci->savedContinueLanesPtr;
        blockEntryMask = ci->savedBlockEntryMask;
        switchExpr = ci->savedSwitchExpr;
        switchFallThroughMaskPtr = ci->savedSwitchFallThroughMaskPtr;
        defaultBlock = ci->savedDefaultBlock;
        caseBlocks = ci->savedCaseBlocks;
        nextBlocks = ci->savedNextBlocks;
        switchConditionWasUniform = ci->savedSwitchConditionWasUniform;
    } else if (ci->IsLoop() || ci->IsForeach()) {
        breakTarget = ci->savedBreakTarget;
        continueTarget = ci->savedContinueTarget;
        breakLanesPtr = ci->savedBreakLanesPtr;
        continueLanesPtr = ci->savedContinueLanesPtr;
        blockEntryMask = ci->savedBlockEntryMask;
    } else {
        AssertPos(currentPos, ci->IsIf());
        // nothing to do
    }

    return ci;
}

#ifdef ISPC_XE_ENABLED
bool FunctionEmitContext::inXeSimdCF() const {
    // Go backwards through controlFlowInfo, since we add new nested scopes
    // to the back.
    if (controlFlowInfo.size() > 0) {
        int i = controlFlowInfo.size() - 1;
        while (i >= 0) {
            if (controlFlowInfo[i]->isUniformEmulated == true)
                // Found a scope due to an 'if' statement with a emulated uniform test
                return true;
            --i;
        }
    }
    return false;
}

llvm::Value *FunctionEmitContext::XeSimdCFAny(llvm::Value *value) {
    AssertPos(currentPos, llvm::isa<llvm::VectorType>(value->getType()));
    llvm::Value *mask = GetInternalMask();
    value = BinaryOperator(llvm::BinaryOperator::And, mask, value);
    auto Fn = llvm::GenXIntrinsic::getGenXDeclaration(m->module, llvm::GenXIntrinsic::genx_simdcf_any,
                                                      LLVMTypes::Int1VectorType);
    return llvm::CallInst::Create(Fn, value, "", bblock);
}

llvm::Value *FunctionEmitContext::XeSimdCFPredicate(llvm::Value *value, llvm::Value *defaults) {
    AssertPos(currentPos, llvm::isa<llvm::VectorType>(value->getType()));
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(value->getType());
#else
    llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(value->getType());
#endif
    if (defaults == NULL) {
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
        defaults = llvm::ConstantVector::getSplat(vt->getVectorNumElements(),
                                                  llvm::Constant::getNullValue(vt->getElementType()));
#elif ISPC_LLVM_VERSION < ISPC_LLVM_12_0
        defaults = llvm::ConstantVector::getSplat({static_cast<unsigned int>(vt->getNumElements()), false},
                                                  llvm::Constant::getNullValue(vt->getElementType()));
#else
        defaults = llvm::ConstantVector::getSplat(
            llvm::ElementCount::get(static_cast<unsigned int>(vt->getNumElements()), false),
            llvm::Constant::getNullValue(vt->getElementType()));
#endif
    }

    auto Fn = llvm::GenXIntrinsic::getGenXDeclaration(m->module, llvm::GenXIntrinsic::genx_simdcf_predicate,
                                                      value->getType());
    std::vector<llvm::Value *> args;
    args.push_back(value);
    args.push_back(defaults);
    return llvm::CallInst::Create(Fn, args, "", bblock);
}

llvm::Value *FunctionEmitContext::XePrepareVectorBranch(llvm::Value *value) {
    llvm::Value *ret = value;
    // If condition is varying we just insert simdcf.any intrinsic.
    // If condition is a scalar we should change it to vector but only if we had
    // varying condition which was emulated as uniform in external scopes.
    if (!llvm::isa<llvm::VectorType>(value->getType())) {
        if (!inXeSimdCF())
            return ret;
        ret = BroadcastValue(value, LLVMTypes::Int1VectorType);
    }
    Assert(ret != NULL);
    return XeSimdCFAny(ret);
}

llvm::Value *FunctionEmitContext::XeStartUnmaskedRegion() {
    auto Fn = llvm::GenXIntrinsic::getGenXDeclaration(m->module, llvm::GenXIntrinsic::genx_unmask_begin);
    std::vector<llvm::Value *> args;
    llvm::Value *maskAlloca = AllocaInst(LLVMTypes::Int32Type);
    llvm::Value *execMask = llvm::CallInst::Create(Fn, args, "", bblock);
    StoreInst(execMask, maskAlloca);
    return maskAlloca;
}

void FunctionEmitContext::XeEndUnmaskedRegion(llvm::Value *execMask) {
    llvm::Value *restoredMask = LoadInst(execMask);
    auto Fn = llvm::GenXIntrinsic::getGenXDeclaration(m->module, llvm::GenXIntrinsic::genx_unmask_end);
    llvm::CallInst::Create(Fn, restoredMask, "", bblock);
}

void FunctionEmitContext::XeUniformMetadata(llvm::Value *v) {
    llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(v);
    // Set ISPC-Uniform to exclude instruction from predication in CMSIMDCFLowering.
    if (inst != NULL) {
        llvm::MDNode *N = llvm::MDNode::get(*g->ctx, llvm::MDString::get(*g->ctx, "ISPC-Uniform"));
        inst->setMetadata("ISPC-Uniform", N);
    }
}

llvm::Constant *FunctionEmitContext::XeCreateConstantString(llvm::StringRef str, llvm::StringRef name) {
    auto *initializer = llvm::ConstantDataArray::getString(*g->ctx, str, /* AddNull */ true);
    auto *GV = new llvm::GlobalVariable(*m->module, initializer->getType(),
                                        /* const */ true, llvm::GlobalValue::InternalLinkage, initializer, name,
                                        nullptr, llvm::GlobalVariable::NotThreadLocal,
                                        /* Constant Addrspace */ 2);
    GV->setAlignment(llvm::MaybeAlign(g->target->getDataLayout()->getABITypeAlignment(initializer->getType())));
    GV->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

    return llvm::ConstantExpr::getInBoundsGetElementPtr(GV->getValueType(), GV,
                                                        llvm::ArrayRef<llvm::Constant *>{LLVMInt32(0), LLVMInt32(0)});
}

llvm::Constant *FunctionEmitContext::XeGetOrCreateConstantString(llvm::StringRef str, llvm::StringRef name) {
    auto *GV = m->module->getGlobalVariable(name, /* AllowInternal */ true);
    if (GV)
        return llvm::ConstantExpr::getInBoundsGetElementPtr(
            GV->getValueType(), GV, llvm::ArrayRef<llvm::Constant *>{LLVMInt32(0), LLVMInt32(0)});
    return XeCreateConstantString(str, name);
}

#endif

bool FunctionEmitContext::emitXeHardwareMask() {
    bool emitXeHardwareMask = g->target->isXeTarget();
#ifdef ISPC_XE_ENABLED
    emitXeHardwareMask &= g->opt.emitXeHardwareMask;
#endif
    return emitXeHardwareMask;
}
} // namespace ispc
