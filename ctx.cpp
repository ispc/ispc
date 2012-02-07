/*
  Copyright (c) 2010-2011, Intel Corporation
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
#include "util.h"
#include "func.h"
#include "llvmutil.h"
#include "type.h"
#include "stmt.h"
#include "expr.h"
#include "module.h"
#include "sym.h"
#include <map>
#include <llvm/DerivedTypes.h>
#include <llvm/Instructions.h>
#include <llvm/Support/Dwarf.h>
#include <llvm/Metadata.h>
#include <llvm/Module.h>

/** This is a small utility structure that records information related to one
    level of nested control flow.  It's mostly used in correctly restoring
    the mask and other state as we exit control flow nesting levels. 
*/
struct CFInfo {
    /** Returns a new instance of the structure that represents entering an
        'if' statement */
    static CFInfo *GetIf(bool isUniform, llvm::Value *savedMask);

    /** Returns a new instance of the structure that represents entering a
        loop. */
    static CFInfo *GetLoop(bool isUniform, llvm::BasicBlock *breakTarget,
                           llvm::BasicBlock *continueTarget, 
                           llvm::Value *savedBreakLanesPtr,
                           llvm::Value *savedContinueLanesPtr,
                           llvm::Value *savedMask, llvm::Value *savedLoopMask);

    static CFInfo *GetForeach(llvm::BasicBlock *breakTarget,
                              llvm::BasicBlock *continueTarget, 
                              llvm::Value *savedBreakLanesPtr,
                              llvm::Value *savedContinueLanesPtr,
                              llvm::Value *savedMask, llvm::Value *savedLoopMask);

    static CFInfo *GetSwitch(bool isUniform, llvm::BasicBlock *breakTarget,
                             llvm::BasicBlock *continueTarget, 
                             llvm::Value *savedBreakLanesPtr,
                             llvm::Value *savedContinueLanesPtr,
                             llvm::Value *savedMask, llvm::Value *savedLoopMask,
                             llvm::Value *switchExpr,
                             llvm::BasicBlock *bbDefault,
                             const std::vector<std::pair<int, llvm::BasicBlock *> > *bbCases,
                             const std::map<llvm::BasicBlock *, llvm::BasicBlock *> *bbNext,
                             bool scUniform);
    
    bool IsIf() { return type == If; }
    bool IsLoop() { return type == Loop; }
    bool IsForeach() { return type == Foreach; }
    bool IsSwitch() { return type == Switch; }
    bool IsVarying() { return !isUniform; }
    bool IsUniform() { return isUniform; }

    enum CFType { If, Loop, Foreach, Switch };
    CFType type;
    bool isUniform;
    llvm::BasicBlock *savedBreakTarget, *savedContinueTarget;
    llvm::Value *savedBreakLanesPtr, *savedContinueLanesPtr;
    llvm::Value *savedMask, *savedLoopMask;
    llvm::Value *savedSwitchExpr;
    llvm::BasicBlock *savedDefaultBlock;
    const std::vector<std::pair<int, llvm::BasicBlock *> > *savedCaseBlocks;
    const std::map<llvm::BasicBlock *, llvm::BasicBlock *> *savedNextBlocks;
    bool savedSwitchConditionWasUniform;

private:
    CFInfo(CFType t, bool uniformIf, llvm::Value *sm) {
        Assert(t == If);
        type = t;
        isUniform = uniformIf;
        savedBreakTarget = savedContinueTarget = NULL;
        savedBreakLanesPtr = savedContinueLanesPtr = NULL;
        savedMask = savedLoopMask = sm;
        savedSwitchExpr = NULL;
        savedDefaultBlock = NULL;
        savedCaseBlocks = NULL;
        savedNextBlocks = NULL;
    }
    CFInfo(CFType t, bool iu, llvm::BasicBlock *bt, llvm::BasicBlock *ct,
           llvm::Value *sb, llvm::Value *sc, llvm::Value *sm,
           llvm::Value *lm, llvm::Value *sse = NULL, llvm::BasicBlock *bbd = NULL, 
           const std::vector<std::pair<int, llvm::BasicBlock *> > *bbc = NULL,
           const std::map<llvm::BasicBlock *, llvm::BasicBlock *> *bbn = NULL,
           bool scu = false) {
        Assert(t == Loop || t == Switch);
        type = t;
        isUniform = iu;
        savedBreakTarget = bt;
        savedContinueTarget = ct;
        savedBreakLanesPtr = sb;
        savedContinueLanesPtr = sc;
        savedMask = sm;
        savedLoopMask = lm;
        savedSwitchExpr = sse;
        savedDefaultBlock = bbd;
        savedCaseBlocks = bbc;
        savedNextBlocks = bbn;
        savedSwitchConditionWasUniform = scu;
    }
    CFInfo(CFType t, llvm::BasicBlock *bt, llvm::BasicBlock *ct,
           llvm::Value *sb, llvm::Value *sc, llvm::Value *sm,
           llvm::Value *lm) {
        Assert(t == Foreach);
        type = t;
        isUniform = false;
        savedBreakTarget = bt;
        savedContinueTarget = ct;
        savedBreakLanesPtr = sb;
        savedContinueLanesPtr = sc;
        savedMask = sm;
        savedLoopMask = lm;
        savedSwitchExpr = NULL;
        savedDefaultBlock = NULL;
        savedCaseBlocks = NULL;
        savedNextBlocks = NULL;
    }
};


CFInfo *
CFInfo::GetIf(bool isUniform, llvm::Value *savedMask) {
    return new CFInfo(If, isUniform, savedMask);
}


CFInfo *
CFInfo::GetLoop(bool isUniform, llvm::BasicBlock *breakTarget,
                llvm::BasicBlock *continueTarget, 
                llvm::Value *savedBreakLanesPtr,
                llvm::Value *savedContinueLanesPtr,
                llvm::Value *savedMask, llvm::Value *savedLoopMask) {
    return new CFInfo(Loop, isUniform, breakTarget, continueTarget,
                      savedBreakLanesPtr, savedContinueLanesPtr,
                      savedMask, savedLoopMask);
}


CFInfo *
CFInfo::GetForeach(llvm::BasicBlock *breakTarget,
                   llvm::BasicBlock *continueTarget, 
                   llvm::Value *savedBreakLanesPtr,
                   llvm::Value *savedContinueLanesPtr,
                   llvm::Value *savedMask, llvm::Value *savedForeachMask) {
    return new CFInfo(Foreach, breakTarget, continueTarget,
                      savedBreakLanesPtr, savedContinueLanesPtr,
                      savedMask, savedForeachMask);
}


CFInfo *
CFInfo::GetSwitch(bool isUniform, llvm::BasicBlock *breakTarget,
                  llvm::BasicBlock *continueTarget, 
                  llvm::Value *savedBreakLanesPtr,
                  llvm::Value *savedContinueLanesPtr, llvm::Value *savedMask,
                  llvm::Value *savedLoopMask, llvm::Value *savedSwitchExpr,
                  llvm::BasicBlock *savedDefaultBlock,
                  const std::vector<std::pair<int, llvm::BasicBlock *> > *savedCases,
                  const std::map<llvm::BasicBlock *, llvm::BasicBlock *> *savedNext,
                  bool savedSwitchConditionUniform) {
    return new CFInfo(Switch, isUniform, breakTarget, continueTarget, 
                      savedBreakLanesPtr, savedContinueLanesPtr,
                      savedMask, savedLoopMask, savedSwitchExpr, savedDefaultBlock, 
                      savedCases, savedNext, savedSwitchConditionUniform);
}

///////////////////////////////////////////////////////////////////////////

FunctionEmitContext::FunctionEmitContext(Function *func, Symbol *funSym,
                                         llvm::Function *lf,
                                         SourcePos firstStmtPos) {
    function = func;
    llvmFunction = lf;

    /* Create a new basic block to store all of the allocas */
    allocaBlock = llvm::BasicBlock::Create(*g->ctx, "allocas", llvmFunction, 0);
    bblock = llvm::BasicBlock::Create(*g->ctx, "entry", llvmFunction, 0);
    /* But jump from it immediately into the real entry block */
    llvm::BranchInst::Create(bblock, allocaBlock);

    funcStartPos = funSym->pos;

    internalMaskPointer = AllocaInst(LLVMTypes::MaskType, "internal_mask_memory");
    StoreInst(LLVMMaskAllOn, internalMaskPointer);

    functionMaskValue = LLVMMaskAllOn;

    fullMaskPointer = AllocaInst(LLVMTypes::MaskType, "full_mask_memory");
    StoreInst(LLVMMaskAllOn, fullMaskPointer);

    loopMask = NULL;
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
    StoreInst(llvm::Constant::getNullValue(LLVMTypes::VoidPointerType), 
              launchGroupHandlePtr);

    const Type *returnType = function->GetReturnType();
    if (!returnType || returnType == AtomicType::Void)
        returnValuePtr = NULL;
    else {
        LLVM_TYPE_CONST llvm::Type *ftype = returnType->LLVMType(g->ctx);
        returnValuePtr = AllocaInst(ftype, "return_value_memory");
    }

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
        llvm::Value *globalAllOnMaskPtr = 
            m->module->getNamedGlobal("__all_on_mask");
        if (globalAllOnMaskPtr == NULL) {
            globalAllOnMaskPtr = 
                new llvm::GlobalVariable(*m->module, LLVMTypes::MaskType, false,
                                         llvm::GlobalValue::InternalLinkage,
                                         LLVMMaskAllOn, "__all_on_mask");

            char buf[256];
            sprintf(buf, "__off_all_on_mask_%s", g->target.GetISAString());
            llvm::Constant *offFunc = 
                m->module->getOrInsertFunction(buf, LLVMTypes::VoidType,
                                               NULL);
            Assert(llvm::isa<llvm::Function>(offFunc));
            llvm::BasicBlock *offBB = 
                   llvm::BasicBlock::Create(*g->ctx, "entry", 
                                            (llvm::Function *)offFunc, 0);
            new llvm::StoreInst(LLVMMaskAllOff, globalAllOnMaskPtr, offBB);
            llvm::ReturnInst::Create(*g->ctx, offBB);
        }

        llvm::Value *allOnMask = LoadInst(globalAllOnMaskPtr, "all_on_mask");
        SetInternalMaskAnd(LLVMMaskAllOn, allOnMask);
    }

    if (m->diBuilder) {
        /* If debugging is enabled, tell the debug information emission
           code about this new function */
        diFile = funcStartPos.GetDIFile();
        llvm::DIType retType = function->GetReturnType()->GetDIType(diFile);
        int flags = llvm::DIDescriptor::FlagPrototyped; // ??
        diFunction = m->diBuilder->createFunction(diFile, /* scope */
                                                  llvmFunction->getName(), // mangled
                                                  funSym->name,
                                                  diFile,
                                                  funcStartPos.first_line,
                                                  retType,
                                                  funSym->storageClass == SC_STATIC,
                                                  true, /* is definition */
                                                  flags,
                                                  g->opt.level > 0,
                                                  llvmFunction);
        /* And start a scope representing the initial function scope */
        StartScope();

        llvm::DIFile file = funcStartPos.GetDIFile();
        Symbol *programIndexSymbol = m->symbolTable->LookupVariable("programIndex");
        Assert(programIndexSymbol && programIndexSymbol->storagePtr);
        m->diBuilder->createGlobalVariable(programIndexSymbol->name, 
                                           file,
                                           funcStartPos.first_line,
                                           programIndexSymbol->type->GetDIType(file),
                                           true /* static */,
                                           programIndexSymbol->storagePtr);

        Symbol *programCountSymbol = m->symbolTable->LookupVariable("programCount");
        Assert(programCountSymbol);
        m->diBuilder->createGlobalVariable(programCountSymbol->name, 
                                           file,
                                           funcStartPos.first_line,
                                           programCountSymbol->type->GetDIType(file),
                                           true /* static */,
                                           programCountSymbol->storagePtr);
    }
}


FunctionEmitContext::~FunctionEmitContext() {
    Assert(controlFlowInfo.size() == 0);
    Assert(debugScopes.size() == (m->diBuilder ? 1 : 0));
}


const Function *
FunctionEmitContext::GetFunction() const {
    return function;
}


llvm::BasicBlock *
FunctionEmitContext::GetCurrentBasicBlock() {
    return bblock;
}


void
FunctionEmitContext::SetCurrentBasicBlock(llvm::BasicBlock *bb) {
    bblock = bb;
}


llvm::Value *
FunctionEmitContext::GetFunctionMask() {
    return functionMaskValue;
}


llvm::Value *
FunctionEmitContext::GetInternalMask() {
    return LoadInst(internalMaskPointer, "load_mask");
}


llvm::Value *
FunctionEmitContext::GetFullMask() {
    llvm::Value *internalMask = GetInternalMask();
    if (internalMask == LLVMMaskAllOn && functionMaskValue == LLVMMaskAllOn &&
        !g->opt.disableMaskAllOnOptimizations)
        return LLVMMaskAllOn;
    else
        return BinaryOperator(llvm::Instruction::And, GetInternalMask(), 
                              functionMaskValue, "internal_mask&function_mask");
}


llvm::Value *
FunctionEmitContext::GetFullMaskPointer() {
    return fullMaskPointer;
}


void
FunctionEmitContext::SetFunctionMask(llvm::Value *value) {
    functionMaskValue = value;
    if (bblock != NULL)
        StoreInst(GetFullMask(), fullMaskPointer);
}


void
FunctionEmitContext::SetLoopMask(llvm::Value *value) {
    loopMask = value;
}


void
FunctionEmitContext::SetInternalMask(llvm::Value *value) {
    StoreInst(value, internalMaskPointer);
    // kludge so that __mask returns the right value in ispc code.
    StoreInst(GetFullMask(), fullMaskPointer);
}


void
FunctionEmitContext::SetInternalMaskAnd(llvm::Value *oldMask, llvm::Value *test) {
    llvm::Value *mask = BinaryOperator(llvm::Instruction::And, oldMask, 
                                       test, "oldMask&test");
    SetInternalMask(mask);
}


void
FunctionEmitContext::SetInternalMaskAndNot(llvm::Value *oldMask, llvm::Value *test) {
    llvm::Value *notTest = BinaryOperator(llvm::Instruction::Xor, test, LLVMMaskAllOn,
                                          "~test");
    llvm::Value *mask = BinaryOperator(llvm::Instruction::And, oldMask, notTest,
                                       "oldMask&~test");
    SetInternalMask(mask);
}


void
FunctionEmitContext::BranchIfMaskAny(llvm::BasicBlock *btrue, llvm::BasicBlock *bfalse) {
    Assert(bblock != NULL);
    llvm::Value *any = Any(GetFullMask());
    BranchInst(btrue, bfalse, any);
    // It's illegal to add any additional instructions to the basic block
    // now that it's terminated, so set bblock to NULL to be safe
    bblock = NULL;
}


void
FunctionEmitContext::BranchIfMaskAll(llvm::BasicBlock *btrue, llvm::BasicBlock *bfalse) {
    Assert(bblock != NULL);
    llvm::Value *all = All(GetFullMask());
    BranchInst(btrue, bfalse, all);
    // It's illegal to add any additional instructions to the basic block
    // now that it's terminated, so set bblock to NULL to be safe
    bblock = NULL;
}


void
FunctionEmitContext::BranchIfMaskNone(llvm::BasicBlock *btrue, llvm::BasicBlock *bfalse) {
    Assert(bblock != NULL);
    // switch sense of true/false bblocks
    BranchIfMaskAny(bfalse, btrue);
    // It's illegal to add any additional instructions to the basic block
    // now that it's terminated, so set bblock to NULL to be safe
    bblock = NULL;
}


void
FunctionEmitContext::StartUniformIf() {
    controlFlowInfo.push_back(CFInfo::GetIf(true, GetInternalMask()));
}


void
FunctionEmitContext::StartVaryingIf(llvm::Value *oldMask) {
    controlFlowInfo.push_back(CFInfo::GetIf(false, oldMask));
}


void
FunctionEmitContext::EndIf() {
    CFInfo *ci = popCFState();
    // Make sure we match up with a Start{Uniform,Varying}If().
    Assert(ci->IsIf());

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
            bcLanes = LoadInst(continueLanesPtr, "continue_lanes");
        else
            bcLanes = LLVMMaskAllOff;

        if (breakLanesPtr != NULL) {
            llvm::Value *breakLanes = LoadInst(breakLanesPtr, "break_lanes");
            bcLanes = BinaryOperator(llvm::Instruction::Or, bcLanes, 
                                     breakLanes, "|break_lanes");
        }

        llvm::Value *notBreakOrContinue = 
            NotOperator(bcLanes, "!(break|continue)_lanes");
        llvm::Value *oldMask = GetInternalMask();
        llvm::Value *newMask = 
            BinaryOperator(llvm::Instruction::And, oldMask, 
                           notBreakOrContinue, "new_mask");
        SetInternalMask(newMask);
    }
}


void
FunctionEmitContext::StartLoop(llvm::BasicBlock *bt, llvm::BasicBlock *ct, 
                               bool uniformCF) {
    // Store the current values of various loop-related state so that we
    // can restore it when we exit this loop.
    llvm::Value *oldMask = GetInternalMask();
    controlFlowInfo.push_back(CFInfo::GetLoop(uniformCF, breakTarget, 
                                              continueTarget, breakLanesPtr,
                                              continueLanesPtr, oldMask, loopMask));
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
    loopMask = NULL; // this better be set by the loop!
}


void
FunctionEmitContext::EndLoop() {
    CFInfo *ci = popCFState();
    Assert(ci->IsLoop());

    if (!ci->IsUniform())
        // If the loop had a 'uniform' test, then it didn't make any
        // changes to the mask so there's nothing to restore.  If it had a
        // varying test, we need to restore the mask to what it was going
        // into the loop, but still leaving off any lanes that executed a
        // 'return' statement.
        restoreMaskGivenReturns(ci->savedMask);
}


void
FunctionEmitContext::StartForeach() {
    // Store the current values of various loop-related state so that we
    // can restore it when we exit this loop.
    llvm::Value *oldMask = GetInternalMask();
    controlFlowInfo.push_back(CFInfo::GetForeach(breakTarget, continueTarget, 
                                                 breakLanesPtr, continueLanesPtr,
                                                 oldMask, loopMask));
    breakLanesPtr = NULL;
    breakTarget = NULL;

    continueLanesPtr = AllocaInst(LLVMTypes::MaskType, "foreach_continue_lanes");
    StoreInst(LLVMMaskAllOff, continueLanesPtr);
    continueTarget = NULL; // should be set by SetContinueTarget()

    loopMask = NULL;
}


void
FunctionEmitContext::EndForeach() {
    CFInfo *ci = popCFState();
    Assert(ci->IsForeach());
}


void
FunctionEmitContext::restoreMaskGivenReturns(llvm::Value *oldMask) {
    if (!bblock)
        return;

    // Restore the mask to the given old mask, but leave off any lanes that
    // executed a return statement.
    // newMask = (oldMask & ~returnedLanes)
    llvm::Value *returnedLanes = LoadInst(returnedLanesPtr,
                                          "returned_lanes");
    llvm::Value *notReturned = NotOperator(returnedLanes, "~returned_lanes");
    llvm::Value *newMask = BinaryOperator(llvm::Instruction::And,
                                          oldMask, notReturned, "new_mask");
    SetInternalMask(newMask);
}


/** Returns "true" if the first enclosing non-if control flow expression is
    a "switch" statement.
*/
bool
FunctionEmitContext::inSwitchStatement() const {
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


void
FunctionEmitContext::Break(bool doCoherenceCheck) {
    if (breakTarget == NULL) {
        Error(currentPos, "\"break\" statement is illegal outside of "
              "for/while/do loops and \"switch\" statements.");
        return;
    }
    Assert(controlFlowInfo.size() > 0);

    if (bblock == NULL)
        return;

    if (inSwitchStatement() == true &&
        switchConditionWasUniform == true && 
        ifsInCFAllUniform(CFInfo::Switch)) {
        // We know that all program instances are executing the break, so
        // just jump to the block immediately after the switch.
        Assert(breakTarget != NULL);
        BranchInst(breakTarget);
        bblock = NULL;
        return;
    }

    // If all of the enclosing 'if' tests in the loop have uniform control
    // flow or if we can tell that the mask is all on, then we can just
    // jump to the break location.
    if (inSwitchStatement() == false && 
        (ifsInCFAllUniform(CFInfo::Loop) || 
         GetInternalMask() == LLVMMaskAllOn)) {
        BranchInst(breakTarget);
        if (ifsInCFAllUniform(CFInfo::Loop) && doCoherenceCheck)
            Warning(currentPos, "Coherent break statement not necessary in "
                    "fully uniform control flow.");
        // Set bblock to NULL since the jump has terminated the basic block
        bblock = NULL;
    }
    else {
        // Varying switch, uniform switch where the 'break' is under
        // varying control flow, or a loop with varying 'if's above the
        // break.  In these cases, we need to update the mask of the lanes
        // that have executed a 'break' statement: 
        // breakLanes = breakLanes | mask
        Assert(breakLanesPtr != NULL);
        llvm::Value *mask = GetInternalMask();
        llvm::Value *breakMask = LoadInst(breakLanesPtr,
                                          "break_mask");
        llvm::Value *newMask = BinaryOperator(llvm::Instruction::Or,
                                              mask, breakMask, "mask|break_mask");
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


void
FunctionEmitContext::Continue(bool doCoherenceCheck) {
    if (!continueTarget) {
        Error(currentPos, "\"continue\" statement illegal outside of "
              "for/while/do/foreach loops.");
        return;
    }
    Assert(controlFlowInfo.size() > 0);

    if (ifsInCFAllUniform(CFInfo::Loop) || GetInternalMask() == LLVMMaskAllOn) {
        // Similarly to 'break' statements, we can immediately jump to the
        // continue target if we're only in 'uniform' control flow within
        // loop or if we can tell that the mask is all on.
        AddInstrumentationPoint("continue: uniform CF, jumped");
        if (ifsInCFAllUniform(CFInfo::Loop) && doCoherenceCheck)
            Warning(currentPos, "Coherent continue statement not necessary in "
                    "fully uniform control flow.");
        BranchInst(continueTarget);
        bblock = NULL;
    }
    else {
        // Otherwise update the stored value of which lanes have 'continue'd.
        // continueLanes = continueLanes | mask
        Assert(continueLanesPtr);
        llvm::Value *mask = GetInternalMask();
        llvm::Value *continueMask = 
            LoadInst(continueLanesPtr, "continue_mask");
        llvm::Value *newMask = 
            BinaryOperator(llvm::Instruction::Or, mask, continueMask,
                           "mask|continueMask");
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
bool
FunctionEmitContext::ifsInCFAllUniform(int type) const {
    Assert(controlFlowInfo.size() > 0);
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
    Assert(i >= 0); // else we didn't find the expected control flow type!
    return true;
}


void
FunctionEmitContext::jumpIfAllLoopLanesAreDone(llvm::BasicBlock *target) {
    llvm::Value *allDone = NULL;
    Assert(continueLanesPtr != NULL);
    if (breakLanesPtr == NULL) {
        // In a foreach loop, break and return are illegal, and
        // breakLanesPtr is NULL.  In this case, the mask is guaranteed to
        // be all on at the start of each iteration, so we only need to
        // check if all lanes have continued..
        llvm::Value *continued = LoadInst(continueLanesPtr,
                                          "continue_lanes");
        allDone = All(continued);
    }
    else {
        // Check to see if (returned lanes | continued lanes | break lanes) is
        // equal to the value of mask at the start of the loop iteration.  If
        // so, everyone is done and we can jump to the given target
        llvm::Value *returned = LoadInst(returnedLanesPtr,
                                         "returned_lanes");
        llvm::Value *continued = LoadInst(continueLanesPtr,
                                          "continue_lanes");
        llvm::Value *breaked = LoadInst(breakLanesPtr, "break_lanes");
        llvm::Value *returnedOrContinued = BinaryOperator(llvm::Instruction::Or, 
                                                          returned, continued,
                                                          "returned|continued");
        llvm::Value *returnedOrContinuedOrBreaked = 
            BinaryOperator(llvm::Instruction::Or, returnedOrContinued,
                           breaked, "returned|continued");

        // Do we match the mask at loop entry?
        allDone = MasksAllEqual(returnedOrContinuedOrBreaked, loopMask);
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


void
FunctionEmitContext::RestoreContinuedLanes() {
    if (continueLanesPtr == NULL)
        return;

    // mask = mask & continueFlags
    llvm::Value *mask = GetInternalMask();
    llvm::Value *continueMask = LoadInst(continueLanesPtr,
                                         "continue_mask");
    llvm::Value *orMask = BinaryOperator(llvm::Instruction::Or,
                                         mask, continueMask, "mask|continue_mask");
    SetInternalMask(orMask);

    // continueLanes = 0
    StoreInst(LLVMMaskAllOff, continueLanesPtr);
}


void
FunctionEmitContext::StartSwitch(bool cfIsUniform, llvm::BasicBlock *bbBreak) {
    llvm::Value *oldMask = GetInternalMask();
    controlFlowInfo.push_back(CFInfo::GetSwitch(cfIsUniform, breakTarget, 
                                                continueTarget, breakLanesPtr,
                                                continueLanesPtr, oldMask, 
                                                loopMask, switchExpr, defaultBlock, 
                                                caseBlocks, nextBlocks,
                                                switchConditionWasUniform));

    breakLanesPtr = AllocaInst(LLVMTypes::MaskType, "break_lanes_memory");
    StoreInst(LLVMMaskAllOff, breakLanesPtr);
    breakTarget = bbBreak;

    continueLanesPtr = NULL;
    continueTarget = NULL;
    loopMask = NULL;

    // These will be set by the SwitchInst() method
    switchExpr = NULL;
    defaultBlock = NULL;
    caseBlocks = NULL;
    nextBlocks = NULL;
}


void
FunctionEmitContext::EndSwitch() {
    Assert(bblock != NULL);

    CFInfo *ci = popCFState();
    if (ci->IsVarying() && bblock != NULL)
        restoreMaskGivenReturns(ci->savedMask);
}


/** Emit code to check for an "all off" mask before the code for a 
    case or default label in a "switch" statement.
 */
void
FunctionEmitContext::addSwitchMaskCheck(llvm::Value *mask) {
    llvm::Value *allOff = None(mask);
    llvm::BasicBlock *bbSome = CreateBasicBlock("case_default_on");

    // Find the basic block for the case or default label immediately after
    // the current one in the switch statement--that's where we want to
    // jump if the mask is all off at this label.
    Assert(nextBlocks->find(bblock) != nextBlocks->end());
    llvm::BasicBlock *bbNext = nextBlocks->find(bblock)->second;

    // Jump to the next one of the mask is all off; otherwise jump to the
    // newly created block that will hold the actual code for this label.
    BranchInst(bbNext, bbSome, allOff);
    SetCurrentBasicBlock(bbSome);
}


/** Returns the execution mask at entry to the first enclosing "switch"
    statement. */
llvm::Value *
FunctionEmitContext::getMaskAtSwitchEntry() {
    Assert(controlFlowInfo.size() > 0);
    int i = controlFlowInfo.size() - 1;
    while (i >= 0 && controlFlowInfo[i]->type != CFInfo::Switch)
        --i;
    Assert(i != -1);
    return controlFlowInfo[i]->savedMask;
}


void
FunctionEmitContext::EmitDefaultLabel(bool checkMask, SourcePos pos) {
    if (inSwitchStatement() == false) {
        Error(pos, "\"default\" label illegal outside of \"switch\" "
              "statement.");
        return;
    }

    // If there's a default label in the switch, a basic block for it
    // should have been provided in the previous call to SwitchInst().
    Assert(defaultBlock != NULL);

    if (bblock != NULL)
        // The previous case in the switch fell through, or we're in a
        // varying switch; terminate the current block with a jump to the
        // block for the code for the default label.
        BranchInst(defaultBlock);
    SetCurrentBasicBlock(defaultBlock);

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
        llvm::Value *valueVec = (switchExpr->getType() == LLVMTypes::Int32VectorType) ?
            LLVMInt32Vector(value) : LLVMInt64Vector(value);
        // TODO: for AVX2 at least, the following generates better code
        // than doing ICMP_NE and skipping the NotOperator() below; file a
        // LLVM bug?
        llvm::Value *matchesCaseValue = 
            CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, switchExpr,
                    valueVec, "cmp_case_value");
        matchesCaseValue = I1VecToBoolVec(matchesCaseValue);

        llvm::Value *notMatchesCaseValue = NotOperator(matchesCaseValue);
        matchesDefault = BinaryOperator(llvm::Instruction::And, matchesDefault, 
                                        notMatchesCaseValue, "default&~case_match");
    }

    // The mask may have some lanes on, which corresponds to the previous
    // label falling through; compute the updated mask by ANDing with the
    // current mask.
    llvm::Value *oldMask = GetInternalMask();
    llvm::Value *newMask = BinaryOperator(llvm::Instruction::Or, oldMask, 
                                          matchesDefault, "old_mask|matches_default");
    SetInternalMask(newMask);

    if (checkMask)
        addSwitchMaskCheck(newMask);
}


void
FunctionEmitContext::EmitCaseLabel(int value, bool checkMask, SourcePos pos) {
    if (inSwitchStatement() == false) {
        Error(pos, "\"case\" label illegal outside of \"switch\" statement.");
        return;
    }

    // Find the basic block for this case statement.
    llvm::BasicBlock *bbCase = NULL;
    Assert(caseBlocks != NULL);
    for (int i = 0; i < (int)caseBlocks->size(); ++i)
        if ((*caseBlocks)[i].first == value) {
            bbCase = (*caseBlocks)[i].second;
            break;
        }
    Assert(bbCase != NULL);

    if (bblock != NULL)
        // fall through from the previous case
        BranchInst(bbCase);
    SetCurrentBasicBlock(bbCase);

    if (switchConditionWasUniform)
        return;

    // update the mask: first, get a mask that indicates which program
    // instances have a value for the switch expression that matches this
    // case statement.
    llvm::Value *valueVec = (switchExpr->getType() == LLVMTypes::Int32VectorType) ?
        LLVMInt32Vector(value) : LLVMInt64Vector(value);
    llvm::Value *matchesCaseValue = 
        CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, switchExpr,
                valueVec, "cmp_case_value");
    matchesCaseValue = I1VecToBoolVec(matchesCaseValue);

    // If a lane was off going into the switch, we don't care if has a
    // value in the switch expression that happens to match this case.
    llvm::Value *entryMask = getMaskAtSwitchEntry();
    matchesCaseValue = BinaryOperator(llvm::Instruction::And, entryMask,
                                      matchesCaseValue, "entry_mask&case_match");

    // Take the surviving lanes and turn on the mask for them.
    llvm::Value *oldMask = GetInternalMask();
    llvm::Value *newMask = BinaryOperator(llvm::Instruction::Or, oldMask, 
                                          matchesCaseValue, "mask|case_match");
    SetInternalMask(newMask);

    if (checkMask)
        addSwitchMaskCheck(newMask);
}


void
FunctionEmitContext::SwitchInst(llvm::Value *expr, llvm::BasicBlock *bbDefault,
                const std::vector<std::pair<int, llvm::BasicBlock *> > &bbCases,
                const std::map<llvm::BasicBlock *, llvm::BasicBlock *> &bbNext) {
    // The calling code should have called StartSwitch() before calling
    // SwitchInst().
    Assert(controlFlowInfo.size() &&
           controlFlowInfo.back()->IsSwitch());

    switchExpr = expr;
    defaultBlock = bbDefault;
    caseBlocks = new std::vector<std::pair<int, llvm::BasicBlock *> >(bbCases);
    nextBlocks = new std::map<llvm::BasicBlock *, llvm::BasicBlock *>(bbNext);
    switchConditionWasUniform = 
        (llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(expr->getType()) == false);

    if (switchConditionWasUniform == true) {
        // For a uniform switch condition, just wire things up to the LLVM
        // switch instruction.
        llvm::SwitchInst *s = llvm::SwitchInst::Create(expr, bbDefault, 
                                                       bbCases.size(), bblock);
        for (int i = 0; i < (int)bbCases.size(); ++i) {
            if (expr->getType() == LLVMTypes::Int32Type)
                s->addCase(LLVMInt32(bbCases[i].first), bbCases[i].second);
            else {
                Assert(expr->getType() == LLVMTypes::Int64Type);
                s->addCase(LLVMInt64(bbCases[i].first), bbCases[i].second);
            }
        }

        AddDebugPos(s);
        // switch is a terminator
        bblock = NULL;
    }
    else {
        // For a varying switch, we first turn off all lanes of the mask
        SetInternalMask(LLVMMaskAllOff);

        if (nextBlocks->size() > 0) {
            // If there are any labels inside the switch, jump to the first
            // one; any code before the first label won't be executed by
            // anyone.
            std::map<llvm::BasicBlock *, llvm::BasicBlock *>::const_iterator iter;
            iter = nextBlocks->find(NULL);
            Assert(iter != nextBlocks->end());
            llvm::BasicBlock *bbFirst = iter->second;
            BranchInst(bbFirst);
            bblock = NULL;
        }
    }
}


int
FunctionEmitContext::VaryingCFDepth() const { 
    int sum = 0;
    for (unsigned int i = 0; i < controlFlowInfo.size(); ++i)
        if (controlFlowInfo[i]->IsVarying())
            ++sum;
    return sum;
}


bool
FunctionEmitContext::InForeachLoop() const {
    for (unsigned int i = 0; i < controlFlowInfo.size(); ++i)
        if (controlFlowInfo[i]->IsForeach())
            return true;
    return false;
}


bool
FunctionEmitContext::initLabelBBlocks(ASTNode *node, void *data) {
    LabeledStmt *ls = dynamic_cast<LabeledStmt *>(node);
    if (ls == NULL)
        return true;

    FunctionEmitContext *ctx = (FunctionEmitContext *)data;

    if (ctx->labelMap.find(ls->name) != ctx->labelMap.end())
        Error(ls->pos, "Multiple labels named \"%s\" in function.",
              ls->name.c_str());
    else {
        llvm::BasicBlock *bb = ctx->CreateBasicBlock(ls->name.c_str());
        ctx->labelMap[ls->name] = bb;
    }
    return true;
}


void
FunctionEmitContext::InitializeLabelMap(Stmt *code) {
    labelMap.erase(labelMap.begin(), labelMap.end());
    WalkAST(code, initLabelBBlocks, NULL, this);
}


llvm::BasicBlock *
FunctionEmitContext::GetLabeledBasicBlock(const std::string &label) {
    if (labelMap.find(label) != labelMap.end())
        return labelMap[label];
    else
        return NULL;
}


void
FunctionEmitContext::CurrentLanesReturned(Expr *expr, bool doCoherenceCheck) {
    const Type *returnType = function->GetReturnType();
    if (returnType == AtomicType::Void) {
        if (expr != NULL)
            Error(expr->pos, "Can't return non-void type \"%s\" from void function.",
                  expr->GetType()->GetString().c_str());
    }
    else {
        if (expr == NULL) {
            Error(funcStartPos, "Must provide return value for return "
                  "statement for non-void function.");
            return;
        }
        
        expr = TypeConvertExpr(expr, returnType, "return statement");
        if (expr != NULL) {
            llvm::Value *retVal = expr->GetValue(this);
            if (retVal != NULL) {
                if (returnType->IsUniformType() ||
                    dynamic_cast<const ReferenceType *>(returnType) != NULL)
                    StoreInst(retVal, returnValuePtr);
                else {
                    // Use a masked store to store the value of the expression
                    // in the return value memory; this preserves the return
                    // values from other lanes that may have executed return
                    // statements previously.
                    StoreInst(retVal, returnValuePtr, GetInternalMask(), 
                              PointerType::GetUniform(returnType));
                }
            }
        }
    }

    if (VaryingCFDepth() == 0) {
        // If there is only uniform control flow between us and the
        // function entry, then it's guaranteed that all lanes are running,
        // so we can just emit a true return instruction
        AddInstrumentationPoint("return: uniform control flow");
        ReturnInst();
    }
    else {
        // Otherwise we update the returnedLanes value by ANDing it with
        // the current lane mask.
        llvm::Value *oldReturnedLanes = 
            LoadInst(returnedLanesPtr, "old_returned_lanes");
        llvm::Value *newReturnedLanes = 
            BinaryOperator(llvm::Instruction::Or, oldReturnedLanes, 
                           GetInternalMask(), "old_mask|returned_lanes");
        
        // For 'coherent' return statements, emit code to check if all
        // lanes have returned
        if (doCoherenceCheck) {
            // if newReturnedLanes == functionMaskValue, get out of here!
            llvm::Value *cmp = MasksAllEqual(functionMaskValue, 
                                             newReturnedLanes);
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


llvm::Value *
FunctionEmitContext::Any(llvm::Value *mask) {
    llvm::Value *mmval = LaneMask(mask);
    return CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE, mmval,
                   LLVMInt32(0), "any_mm_cmp");
}


llvm::Value *
FunctionEmitContext::All(llvm::Value *mask) {
    llvm::Value *mmval = LaneMask(mask);
    return CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, mmval,
                   LLVMInt32((1<<g->target.vectorWidth)-1), "all_mm_cmp");
}


llvm::Value *
FunctionEmitContext::None(llvm::Value *mask) {
    llvm::Value *mmval = LaneMask(mask);
    return CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, mmval,
                   LLVMInt32(0), "none_mm_cmp");
}


llvm::Value *
FunctionEmitContext::LaneMask(llvm::Value *v) {
    // Call the target-dependent movmsk function to turn the vector mask
    // into an i32 value
    std::vector<Symbol *> mm;
    m->symbolTable->LookupFunction("__movmsk", &mm);
    if (g->target.maskBitCount == 1)
        Assert(mm.size() == 1);
    else
        // There should be one with signed int signature, one unsigned int.
        Assert(mm.size() == 2); 
    // We can actually call either one, since both are i32s as far as
    // LLVM's type system is concerned...
    llvm::Function *fmm = mm[0]->function;
    return CallInst(fmm, NULL, v, "val_movmsk");
}


llvm::Value *
FunctionEmitContext::MasksAllEqual(llvm::Value *v1, llvm::Value *v2) {
#if 0
    // Compare the two masks to get a vector of i1s
    llvm::Value *cmp = CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ,
                               v1, v2, "v1==v2");
    // Turn that into a bool vector type (often i32s)
    cmp = I1VecToBoolVec(cmp);
    // And see if it's all on
    return All(cmp);
#else
    llvm::Value *mm1 = LaneMask(v1);
    llvm::Value *mm2 = LaneMask(v2);
    return CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, mm1, mm2,
                   "v1==v2");
#endif
}


llvm::Value *
FunctionEmitContext::GetStringPtr(const std::string &str) {
#ifdef LLVM_3_1svn
    llvm::Constant *lstr = llvm::ConstantDataArray::getString(*g->ctx, str);
#else
    llvm::Constant *lstr = llvm::ConstantArray::get(*g->ctx, str);
#endif
    llvm::GlobalValue::LinkageTypes linkage = llvm::GlobalValue::InternalLinkage;
    llvm::Value *lstrPtr = new llvm::GlobalVariable(*m->module, lstr->getType(),
                                                    true /*isConst*/, 
                                                    linkage, lstr, "__str");
    return new llvm::BitCastInst(lstrPtr, LLVMTypes::VoidPointerType, 
                                 "str_void_ptr", bblock);
}


llvm::BasicBlock *
FunctionEmitContext::CreateBasicBlock(const char *name) {
    return llvm::BasicBlock::Create(*g->ctx, name, llvmFunction);
}


llvm::Value *
FunctionEmitContext::I1VecToBoolVec(llvm::Value *b) {
    if (b == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    if (g->target.maskBitCount == 1)
        return b;

    LLVM_TYPE_CONST llvm::ArrayType *at = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::ArrayType>(b->getType());
    if (at) {
        // If we're given an array of vectors of i1s, then do the
        // conversion for each of the elements
        LLVM_TYPE_CONST llvm::Type *boolArrayType = 
            llvm::ArrayType::get(LLVMTypes::BoolVectorType, at->getNumElements());
        llvm::Value *ret = llvm::UndefValue::get(boolArrayType);

        for (unsigned int i = 0; i < at->getNumElements(); ++i) {
            llvm::Value *elt = ExtractInst(b, i);
            llvm::Value *sext = SExtInst(elt, LLVMTypes::BoolVectorType, 
                                         "val_to_boolvec32");
            ret = InsertInst(ret, sext, i);
        }
        return ret;
    }
    else
        return SExtInst(b, LLVMTypes::BoolVectorType, "val_to_boolvec32");
}


static llvm::Value *
lGetStringAsValue(llvm::BasicBlock *bblock, const char *s) {
#ifdef LLVM_3_1svn
    llvm::Constant *sConstant = llvm::ConstantDataArray::getString(*g->ctx, s);
#else
    llvm::Constant *sConstant = llvm::ConstantArray::get(*g->ctx, s);
#endif
    llvm::Value *sPtr = new llvm::GlobalVariable(*m->module, sConstant->getType(), 
                                                 true /* const */,
                                                 llvm::GlobalValue::InternalLinkage,
                                                 sConstant, s);
    llvm::Value *indices[2] = { LLVMInt32(0), LLVMInt32(0) };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
    llvm::ArrayRef<llvm::Value *> arrayRef(&indices[0], &indices[2]);
    return llvm::GetElementPtrInst::Create(sPtr, arrayRef, "sptr", bblock);
#else
    return llvm::GetElementPtrInst::Create(sPtr, &indices[0], &indices[2],
                                           "sptr", bblock);
#endif
}


void
FunctionEmitContext::AddInstrumentationPoint(const char *note) {
    Assert(note != NULL);
    if (!g->emitInstrumentation)
        return;

    std::vector<llvm::Value *> args;
    // arg 1: filename as string
    args.push_back(lGetStringAsValue(bblock, currentPos.name));
    // arg 2: provided note
    args.push_back(lGetStringAsValue(bblock, note));
    // arg 3: line number
    args.push_back(LLVMInt32(currentPos.first_line));
    // arg 4: current mask, movmsk'ed down to an int32
    args.push_back(LaneMask(GetFullMask()));

    llvm::Function *finst = m->module->getFunction("ISPCInstrument");
    CallInst(finst, NULL, args, "");
}


void
FunctionEmitContext::SetDebugPos(SourcePos pos) { 
    currentPos = pos; 
}


SourcePos
FunctionEmitContext::GetDebugPos() const {
    return currentPos;
}


void
FunctionEmitContext::AddDebugPos(llvm::Value *value, const SourcePos *pos, 
                                 llvm::DIScope *scope) {
    llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(value);
    if (inst != NULL && m->diBuilder) {
        SourcePos p = pos ? *pos : currentPos;
        if (p.first_line != 0)
            // If first_line == 0, then we're in the middle of setting up
            // the standard library or the like; don't add debug positions
            // for those functions
            inst->setDebugLoc(llvm::DebugLoc::get(p.first_line, p.first_column, 
                                                  scope ? *scope : GetDIScope()));
    }
}


void
FunctionEmitContext::StartScope() {
    if (m->diBuilder != NULL) {
        llvm::DIScope parentScope;
        if (debugScopes.size() > 0)
            parentScope = debugScopes.back();
        else
            parentScope = diFunction;

        llvm::DILexicalBlock lexicalBlock = 
            m->diBuilder->createLexicalBlock(parentScope, diFile,
                                             currentPos.first_line,
                                             currentPos.first_column);
        debugScopes.push_back(lexicalBlock);
    }
}


void
FunctionEmitContext::EndScope() {
    if (m->diBuilder != NULL) {
        Assert(debugScopes.size() > 0);
        debugScopes.pop_back();
    }
}


llvm::DIScope 
FunctionEmitContext::GetDIScope() const {
    Assert(debugScopes.size() > 0);
    return debugScopes.back();
}


void
FunctionEmitContext::EmitVariableDebugInfo(Symbol *sym) {
    if (m->diBuilder == NULL)
        return;

    llvm::DIScope scope = GetDIScope();
    llvm::DIVariable var = 
        m->diBuilder->createLocalVariable(llvm::dwarf::DW_TAG_auto_variable,
                                          scope,
                                          sym->name,
                                          sym->pos.GetDIFile(),
                                          sym->pos.first_line,
                                          sym->type->GetDIType(scope),
                                          true /* preserve through opts */);
    llvm::Instruction *declareInst = 
        m->diBuilder->insertDeclare(sym->storagePtr, var, bblock);
    AddDebugPos(declareInst, &sym->pos, &scope);
}


void
FunctionEmitContext::EmitFunctionParameterDebugInfo(Symbol *sym) {
    if (m->diBuilder == NULL)
        return;

    llvm::DIScope scope = diFunction;
    llvm::DIVariable var = 
        m->diBuilder->createLocalVariable(llvm::dwarf::DW_TAG_arg_variable,
                                          scope,
                                          sym->name,
                                          sym->pos.GetDIFile(),
                                          sym->pos.first_line,
                                          sym->type->GetDIType(scope),
                                          true /* preserve through opts */);
    llvm::Instruction *declareInst = 
        m->diBuilder->insertDeclare(sym->storagePtr, var, bblock);
    AddDebugPos(declareInst, &sym->pos, &scope);
}


/** If the given type is an array of vector types, then it's the
    representation of an ispc VectorType with varying elements.  If it is
    one of these, return the array size (i.e. the VectorType's size).
    Otherwise return zero.
 */
static int
lArrayVectorWidth(LLVM_TYPE_CONST llvm::Type *t) {
    LLVM_TYPE_CONST llvm::ArrayType *arrayType = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::ArrayType>(t);
    if (arrayType == NULL)
        return 0;

    // We shouldn't be seeing arrays of anything but vectors being passed
    // to things like FunctionEmitContext::BinaryOperator() as operands.
    LLVM_TYPE_CONST llvm::VectorType *vectorElementType = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::VectorType>(arrayType->getElementType());
    Assert((vectorElementType != NULL &&
            (int)vectorElementType->getNumElements() == g->target.vectorWidth));
           
    return (int)arrayType->getNumElements();
}


llvm::Value *
FunctionEmitContext::BinaryOperator(llvm::Instruction::BinaryOps inst, 
                                    llvm::Value *v0, llvm::Value *v1, 
                                    const char *name) {
    if (v0 == NULL || v1 == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    Assert(v0->getType() == v1->getType());
    LLVM_TYPE_CONST llvm::Type *type = v0->getType();
    int arraySize = lArrayVectorWidth(type);
    if (arraySize == 0) {
        llvm::Instruction *bop = 
            llvm::BinaryOperator::Create(inst, v0, v1, name ? name : "", bblock);
        AddDebugPos(bop);
        return bop;
    }
    else {
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


llvm::Value *
FunctionEmitContext::NotOperator(llvm::Value *v, const char *name) {
    if (v == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    // Similarly to BinaryOperator, do the operation on all the elements of
    // the array if we're given an array type; otherwise just do the
    // regular llvm operation.
    LLVM_TYPE_CONST llvm::Type *type = v->getType();
    int arraySize = lArrayVectorWidth(type);
    if (arraySize == 0) {
        llvm::Instruction *binst = 
            llvm::BinaryOperator::CreateNot(v, name ? name : "not", bblock);
        AddDebugPos(binst);
        return binst;
    }
    else {
        llvm::Value *ret = llvm::UndefValue::get(type);
        for (int i = 0; i < arraySize; ++i) {
            llvm::Value *a = ExtractInst(v, i);
            llvm::Value *op = 
                llvm::BinaryOperator::CreateNot(a, name ? name : "not", bblock);
            AddDebugPos(op);
            ret = InsertInst(ret, op, i);
        }
        return ret;
    }
}


// Given the llvm Type that represents an ispc VectorType, return an
// equally-shaped type with boolean elements.  (This is the type that will
// be returned from CmpInst with ispc VectorTypes).
static LLVM_TYPE_CONST llvm::Type *
lGetMatchingBoolVectorType(LLVM_TYPE_CONST llvm::Type *type) {
    LLVM_TYPE_CONST llvm::ArrayType *arrayType = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::ArrayType>(type);
    Assert(arrayType != NULL);

    LLVM_TYPE_CONST llvm::VectorType *vectorElementType = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::VectorType>(arrayType->getElementType());
    Assert(vectorElementType != NULL);
    Assert((int)vectorElementType->getNumElements() == g->target.vectorWidth);

    LLVM_TYPE_CONST llvm::Type *base = 
        llvm::VectorType::get(LLVMTypes::BoolType, g->target.vectorWidth);
    return llvm::ArrayType::get(base, arrayType->getNumElements());
}


llvm::Value *
FunctionEmitContext::CmpInst(llvm::Instruction::OtherOps inst, 
                             llvm::CmpInst::Predicate pred,
                             llvm::Value *v0, llvm::Value *v1, 
                             const char *name) {
    if (v0 == NULL || v1 == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    Assert(v0->getType() == v1->getType());
    LLVM_TYPE_CONST llvm::Type *type = v0->getType();
    int arraySize = lArrayVectorWidth(type);
    if (arraySize == 0) {
        llvm::Instruction *ci = 
            llvm::CmpInst::Create(inst, pred, v0, v1, name ? name : "cmp", 
                                  bblock);
        AddDebugPos(ci);
        return ci;
    }
    else {
        LLVM_TYPE_CONST llvm::Type *boolType = lGetMatchingBoolVectorType(type);
        llvm::Value *ret = llvm::UndefValue::get(boolType);
        for (int i = 0; i < arraySize; ++i) {
            llvm::Value *a = ExtractInst(v0, i);
            llvm::Value *b = ExtractInst(v1, i);
            llvm::Value *op = CmpInst(inst, pred, a, b, name);
            ret = InsertInst(ret, op, i);
        }
        return ret;
    }
}


llvm::Value *
FunctionEmitContext::SmearUniform(llvm::Value *value, const char *name) {
    if (value == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    llvm::Value *ret = NULL;
    LLVM_TYPE_CONST llvm::Type *eltType = value->getType();

    LLVM_TYPE_CONST llvm::PointerType *pt = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::PointerType>(eltType);
    if (pt != NULL) {
        // Varying pointers are represented as vectors of i32/i64s
        ret = llvm::UndefValue::get(LLVMTypes::VoidPointerVectorType);
        value = PtrToIntInst(value);
    }
    else
        // All other varying types are represented as vectors of the
        // underlying type.
        ret = llvm::UndefValue::get(llvm::VectorType::get(eltType,
                                                          g->target.vectorWidth));

    for (int i = 0; i < g->target.vectorWidth; ++i) {
        llvm::Twine n = llvm::Twine("smear.") + llvm::Twine(name ? name : "") + 
            llvm::Twine(i);
        ret = InsertInst(ret, value, i, n.str().c_str());
    }

    return ret;
}
                                    

llvm::Value *
FunctionEmitContext::BitCastInst(llvm::Value *value, 
                                 LLVM_TYPE_CONST llvm::Type *type, 
                                 const char *name) {
    if (value == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    llvm::Instruction *inst = 
        new llvm::BitCastInst(value, type, name ? name : "bitcast", bblock);
    AddDebugPos(inst);
    return inst;
}


llvm::Value *
FunctionEmitContext::PtrToIntInst(llvm::Value *value, const char *name) {
    if (value == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    if (llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(value->getType()))
        // no-op for varying pointers; they're already vectors of ints
        return value;

    LLVM_TYPE_CONST llvm::Type *type = LLVMTypes::PointerIntType;
    llvm::Instruction *inst = 
        new llvm::PtrToIntInst(value, type, name ? name : "ptr2int", bblock);
    AddDebugPos(inst);
    return inst;
}


llvm::Value *
FunctionEmitContext::PtrToIntInst(llvm::Value *value, 
                                  LLVM_TYPE_CONST llvm::Type *toType,
                                  const char *name) {
    if (value == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    LLVM_TYPE_CONST llvm::Type *fromType = value->getType();
    if (llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(fromType)) {
        // varying pointer
        if (fromType == toType)
            // already the right type--done
            return value;
        else if (fromType->getScalarSizeInBits() > toType->getScalarSizeInBits())
            return TruncInst(value, toType, "ptr_to_int");
        else {
            Assert(fromType->getScalarSizeInBits() <
                   toType->getScalarSizeInBits());
            return ZExtInst(value, toType, "ptr_to_int");
        }
    }

    llvm::Instruction *inst = 
        new llvm::PtrToIntInst(value, toType, name ? name : "ptr2int", bblock);
    AddDebugPos(inst);
    return inst;
}


llvm::Value *
FunctionEmitContext::IntToPtrInst(llvm::Value *value, 
                                  LLVM_TYPE_CONST llvm::Type *toType,
                                  const char *name) {
    if (value == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    LLVM_TYPE_CONST llvm::Type *fromType = value->getType();
    if (llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(fromType)) {
        // varying pointer
        if (fromType == toType)
            // done
            return value;
        else if (fromType->getScalarSizeInBits() > toType->getScalarSizeInBits())
            return TruncInst(value, toType, "int_to_ptr");
        else {
            Assert(fromType->getScalarSizeInBits() <
                   toType->getScalarSizeInBits());
            return ZExtInst(value, toType, "int_to_ptr");
        }
    }

    llvm::Instruction *inst = 
        new llvm::IntToPtrInst(value, toType, name ? name : "int2ptr", bblock);
    AddDebugPos(inst);
    return inst;
}


llvm::Instruction *
FunctionEmitContext::TruncInst(llvm::Value *value, LLVM_TYPE_CONST llvm::Type *type,
                               const char *name) {
    if (value == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    // TODO: we should probably handle the array case as in
    // e.g. BitCastInst(), but we don't currently need that functionality
    llvm::Instruction *inst = 
        new llvm::TruncInst(value, type, name ? name : "trunc", bblock);
    AddDebugPos(inst);
    return inst;
}


llvm::Instruction *
FunctionEmitContext::CastInst(llvm::Instruction::CastOps op, llvm::Value *value,
                              LLVM_TYPE_CONST llvm::Type *type, const char *name) {
    if (value == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    // TODO: we should probably handle the array case as in
    // e.g. BitCastInst(), but we don't currently need that functionality
    llvm::Instruction *inst = 
        llvm::CastInst::Create(op, value, type, name ? name : "cast", bblock);
    AddDebugPos(inst);
    return inst;
}


llvm::Instruction *
FunctionEmitContext::FPCastInst(llvm::Value *value, LLVM_TYPE_CONST llvm::Type *type, 
                                const char *name) {
    if (value == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    // TODO: we should probably handle the array case as in
    // e.g. BitCastInst(), but we don't currently need that functionality
    llvm::Instruction *inst = 
        llvm::CastInst::CreateFPCast(value, type, name ? name : "fpcast", bblock);
    AddDebugPos(inst);
    return inst;
}


llvm::Instruction *
FunctionEmitContext::SExtInst(llvm::Value *value, LLVM_TYPE_CONST llvm::Type *type, 
                              const char *name) {
    if (value == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    // TODO: we should probably handle the array case as in
    // e.g. BitCastInst(), but we don't currently need that functionality
    llvm::Instruction *inst = 
        new llvm::SExtInst(value, type, name ? name : "sext", bblock);
    AddDebugPos(inst);
    return inst;
}


llvm::Instruction *
FunctionEmitContext::ZExtInst(llvm::Value *value, LLVM_TYPE_CONST llvm::Type *type, 
                              const char *name) {
    if (value == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    // TODO: we should probably handle the array case as in
    // e.g. BitCastInst(), but we don't currently need that functionality
    llvm::Instruction *inst = 
        new llvm::ZExtInst(value, type, name ? name : "zext", bblock);
    AddDebugPos(inst);
    return inst;
}


/** Utility routine used by the GetElementPtrInst() methods; given a
    pointer to some type (either uniform or varying) and an index (also
    either uniform or varying), this returns the new pointer (varying if
    appropriate) given by offsetting the base pointer by the index times
    the size of the object that the pointer points to.
 */
llvm::Value *
FunctionEmitContext::applyVaryingGEP(llvm::Value *basePtr, llvm::Value *index, 
                                     const Type *ptrType) {
    // Find the scale factor for the index (i.e. the size of the object
    // that the pointer(s) point(s) to.
    const Type *scaleType = ptrType->GetBaseType();
    llvm::Value *scale = g->target.SizeOf(scaleType->LLVMType(g->ctx), bblock);

    bool indexIsVarying = 
        llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(index->getType());
    llvm::Value *offset = NULL;
    if (indexIsVarying == false) {
        // Truncate or sign extend the index as appropriate to a 32 or
        // 64-bit type.
        if ((g->target.is32Bit || g->opt.force32BitAddressing) && 
            index->getType() == LLVMTypes::Int64Type)
            index = TruncInst(index, LLVMTypes::Int32Type, "trunc_index");
        else if ((!g->target.is32Bit && !g->opt.force32BitAddressing) &&
                 index->getType() == LLVMTypes::Int32Type)
            index = SExtInst(index, LLVMTypes::Int64Type, "sext_index");

        // do a scalar multiply to get the offset as index * scale and then
        // smear the result out to be a vector; this is more efficient than
        // first promoting both the scale and the index to vectors and then
        // multiplying.
        offset = BinaryOperator(llvm::Instruction::Mul, scale, index);
        offset = SmearUniform(offset, "offset_smear");
    }
    else {
        // Similarly, truncate or sign extend the index to be a 32 or 64
        // bit vector type
        if ((g->target.is32Bit || g->opt.force32BitAddressing) && 
            index->getType() == LLVMTypes::Int64VectorType)
            index = TruncInst(index, LLVMTypes::Int32VectorType, "trunc_index");
        else if ((!g->target.is32Bit && !g->opt.force32BitAddressing) &&
                 index->getType() == LLVMTypes::Int32VectorType)
            index = SExtInst(index, LLVMTypes::Int64VectorType, "sext_index");

        scale = SmearUniform(scale, "scale_smear");

        // offset = index * scale
        offset = BinaryOperator(llvm::Instruction::Mul, scale, index, "offset");
    }

    // For 64-bit targets, if we've been doing our offset calculations in
    // 32 bits, we still have to convert to a 64-bit value before we
    // actually add the offset to the pointer.
    if (g->target.is32Bit == false && g->opt.force32BitAddressing == true)
        offset = SExtInst(offset, LLVMTypes::Int64VectorType, "offset_to_64");

    // Smear out the pointer to be varying; either the base pointer or the
    // index must be varying for this method to be called.
    bool baseIsUniform = 
        (llvm::isa<LLVM_TYPE_CONST llvm::PointerType>(basePtr->getType()));
    Assert(baseIsUniform == false || indexIsVarying == true);
    llvm::Value *varyingPtr = baseIsUniform ? 
        SmearUniform(basePtr, "ptr_smear") : basePtr;

    // newPtr = ptr + offset
    return BinaryOperator(llvm::Instruction::Add, varyingPtr, offset, "new_ptr");
}


llvm::Value *
FunctionEmitContext::GetElementPtrInst(llvm::Value *basePtr, llvm::Value *index, 
                                       const Type *ptrType, const char *name) {
    if (basePtr == NULL || index == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    if (dynamic_cast<const ReferenceType *>(ptrType) != NULL)
        ptrType = PointerType::GetUniform(ptrType->GetReferenceTarget());
    Assert(dynamic_cast<const PointerType *>(ptrType) != NULL);

    bool indexIsVaryingType = 
        llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(index->getType());

    if (indexIsVaryingType == false && ptrType->IsUniformType() == true) {
        // The easy case: both the base pointer and the indices are
        // uniform, so just emit the regular LLVM GEP instruction
        llvm::Value *ind[1] = { index };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
        llvm::ArrayRef<llvm::Value *> arrayRef(&ind[0], &ind[1]);
        llvm::Instruction *inst = 
            llvm::GetElementPtrInst::Create(basePtr, arrayRef,
                                            name ? name : "gep", bblock);
#else
        llvm::Instruction *inst = 
            llvm::GetElementPtrInst::Create(basePtr, &ind[0], &ind[1], 
                                            name ? name : "gep", bblock);
#endif
        AddDebugPos(inst);
        return inst;
    }
    else
        return applyVaryingGEP(basePtr, index, ptrType);
}


llvm::Value *
FunctionEmitContext::GetElementPtrInst(llvm::Value *basePtr, llvm::Value *index0, 
                                       llvm::Value *index1, const Type *ptrType,
                                       const char *name) {
    if (basePtr == NULL || index0 == NULL || index1 == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    if (dynamic_cast<const ReferenceType *>(ptrType) != NULL)
        ptrType = PointerType::GetUniform(ptrType->GetReferenceTarget());
    Assert(dynamic_cast<const PointerType *>(ptrType) != NULL);

    bool index0IsVaryingType = 
        llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(index0->getType());
    bool index1IsVaryingType = 
        llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(index1->getType());

    if (index0IsVaryingType == false && index1IsVaryingType == false && 
        ptrType->IsUniformType() == true) {
        // The easy case: both the base pointer and the indices are
        // uniform, so just emit the regular LLVM GEP instruction
        llvm::Value *indices[2] = { index0, index1 };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
        llvm::ArrayRef<llvm::Value *> arrayRef(&indices[0], &indices[2]);
        llvm::Instruction *inst = 
            llvm::GetElementPtrInst::Create(basePtr, arrayRef,
                                            name ? name : "gep", bblock);
#else
        llvm::Instruction *inst = 
            llvm::GetElementPtrInst::Create(basePtr, &indices[0], &indices[2], 
                                            name ? name : "gep", bblock);
#endif
        AddDebugPos(inst);
        return inst;
    }
    else {
        // Handle the first dimension with index0
        llvm::Value *ptr0 = GetElementPtrInst(basePtr, index0, ptrType);

        // Now index into the second dimension with index1.  First figure
        // out the type of ptr0.
        const Type *baseType = ptrType->GetBaseType();
        const SequentialType *st = dynamic_cast<const SequentialType *>(baseType);
        Assert(st != NULL);

        bool ptr0IsUniform = 
            llvm::isa<LLVM_TYPE_CONST llvm::PointerType>(ptr0->getType());
        const Type *ptr0BaseType = st->GetElementType();
        const Type *ptr0Type = ptr0IsUniform ?
            PointerType::GetUniform(ptr0BaseType) : 
            PointerType::GetVarying(ptr0BaseType);

        return applyVaryingGEP(ptr0, index1, ptr0Type);
    }
}


llvm::Value *
FunctionEmitContext::AddElementOffset(llvm::Value *basePtr, int elementNum,
                                      const Type *ptrType, const char *name) {
    if (ptrType == NULL || ptrType->IsUniformType() ||
        dynamic_cast<const ReferenceType *>(ptrType) != NULL) {
        // If the pointer is uniform or we have a reference (which is a
        // uniform pointer in the end), we can use the regular LLVM GEP.
        llvm::Value *offsets[2] = { LLVMInt32(0), LLVMInt32(elementNum) };
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
        llvm::ArrayRef<llvm::Value *> arrayRef(&offsets[0], &offsets[2]);
        return llvm::GetElementPtrInst::Create(basePtr, arrayRef,
                                               name ? name : "struct_offset", bblock);
#else
        return llvm::GetElementPtrInst::Create(basePtr, &offsets[0], &offsets[2],
                                               name ? name : "struct_offset", bblock);
#endif

    }

    if (dynamic_cast<const ReferenceType *>(ptrType) != NULL)
        ptrType = PointerType::GetUniform(ptrType->GetReferenceTarget());
    Assert(dynamic_cast<const PointerType *>(ptrType) != NULL);

    // Otherwise do the math to find the offset and add it to the given
    // varying pointers
    const StructType *st = 
        dynamic_cast<const StructType *>(ptrType->GetBaseType());
    llvm::Value *offset = NULL;
    if (st != NULL)
        // If the pointer is to a structure, Target::StructOffset() gives
        // us the offset in bytes to the given element of the structure
        offset = g->target.StructOffset(st->LLVMType(g->ctx), elementNum,
                                        bblock);
    else {
        // Otherwise we should have a vector or array here and the offset
        // is given by the element number times the size of the element
        // type of the vector.
        const SequentialType *st = 
            dynamic_cast<const SequentialType *>(ptrType->GetBaseType());
        Assert(st != NULL);
        llvm::Value *size = 
            g->target.SizeOf(st->GetElementType()->LLVMType(g->ctx), bblock);
        llvm::Value *scale = (g->target.is32Bit || g->opt.force32BitAddressing) ?
            LLVMInt32(elementNum) : LLVMInt64(elementNum);
        offset = BinaryOperator(llvm::Instruction::Mul, size, scale);
    }

    offset = SmearUniform(offset, "offset_smear");

    if (g->target.is32Bit == false && g->opt.force32BitAddressing == true)
        // If we're doing 32 bit addressing with a 64 bit target, although
        // we did the math above in 32 bit, we need to go to 64 bit before
        // we add the offset to the varying pointers.
        offset = SExtInst(offset, LLVMTypes::Int64VectorType, "offset_to_64");

    return BinaryOperator(llvm::Instruction::Add, basePtr, offset, 
                          "struct_ptr_offset");
}
    

llvm::Value *
FunctionEmitContext::LoadInst(llvm::Value *ptr, const char *name) {
    if (ptr == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    LLVM_TYPE_CONST llvm::PointerType *pt = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::PointerType>(ptr->getType());
    Assert(pt != NULL);

    // FIXME: it's not clear to me that we generate unaligned vector loads
    // of varying stuff out of the front-end any more.  (Only by the
    // optimization passes that lower gathers to vector loads, I think..)
    // So remove this??
    int align = 0;
    if (llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(pt->getElementType()))
        align = 1;
    llvm::Instruction *inst = new llvm::LoadInst(ptr, name ? name : "load",
                                                 false /* not volatile */,
                                                 align, bblock);
    AddDebugPos(inst);
    return inst;
}


llvm::Value *
FunctionEmitContext::LoadInst(llvm::Value *ptr, llvm::Value *mask,
                              const Type *ptrType, const char *name) {
    if (ptr == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    Assert(ptrType != NULL && mask != NULL);

    if (dynamic_cast<const ReferenceType *>(ptrType) != NULL)
        ptrType = PointerType::GetUniform(ptrType->GetReferenceTarget());

    Assert(dynamic_cast<const PointerType *>(ptrType) != NULL);

    if (ptrType->IsUniformType()) {
        // FIXME: same issue as above load inst regarding alignment...
        //
        // If the ptr is a straight up regular pointer, then just issue
        // a regular load.  First figure out the alignment; in general we
        // can just assume the natural alignment (0 here), but for varying
        // atomic types, we need to make sure that the compiler emits
        // unaligned vector loads, so we specify a reduced alignment here.
        int align = 0;
        const AtomicType *atomicType = 
            dynamic_cast<const AtomicType *>(ptrType->GetBaseType());
        if (atomicType != NULL && atomicType->IsVaryingType())
            // We actually just want to align to the vector element
            // alignment, but can't easily get that here, so just tell LLVM
            // it's totally unaligned.  (This shouldn't make any difference
            // vs the proper alignment in practice.)
            align = 1;
        llvm::Instruction *inst = new llvm::LoadInst(ptr, name ? name : "load",
                                                     false /* not volatile */,
                                                     align, bblock);
        AddDebugPos(inst);
        return inst;
    }
    else {
        // Otherwise we should have a varying ptr and it's time for a
        // gather.
        return gather(ptr, ptrType, GetFullMask(), name);
    }
}


llvm::Value *
FunctionEmitContext::gather(llvm::Value *ptr, const Type *ptrType, 
                            llvm::Value *mask, const char *name) {
    // We should have a varying lvalue if we get here...
    Assert(ptrType->IsVaryingType() &&
           ptr->getType() == LLVMTypes::VoidPointerVectorType);

    const Type *returnType = ptrType->GetBaseType()->GetAsVaryingType();
    LLVM_TYPE_CONST llvm::Type *llvmReturnType = returnType->LLVMType(g->ctx);

    const CollectionType *collectionType = 
        dynamic_cast<const CollectionType *>(ptrType->GetBaseType());
    if (collectionType != NULL) {
        // For collections, recursively gather element wise to find the
        // result.
        llvm::Value *retValue = llvm::UndefValue::get(llvmReturnType);
        for (int i = 0; i < collectionType->GetElementCount(); ++i) {
            llvm::Value *eltPtr = AddElementOffset(ptr, i, ptrType);
            const Type *eltPtrType = 
                PointerType::GetVarying(collectionType->GetElementType(i));
            eltPtr = addVaryingOffsetsIfNeeded(eltPtr, eltPtrType);

            // This in turn will be another gather
            llvm::Value *eltValues = LoadInst(eltPtr, mask, eltPtrType, name);

            retValue = InsertInst(retValue, eltValues, i, "set_value");
        }
        return retValue;
    }
    
    // Otherwise we should just have a basic scalar or pointer type and we
    // can go and do the actual gather
    AddInstrumentationPoint("gather");

    // Figure out which gather function to call based on the size of
    // the elements.
    const PointerType *pt = dynamic_cast<const PointerType *>(returnType);
    const char *funcName = NULL;
    if (pt != NULL)
        funcName = g->target.is32Bit ? "__pseudo_gather32_32" : 
            "__pseudo_gather64_64";
    else if (llvmReturnType == LLVMTypes::DoubleVectorType || 
             llvmReturnType == LLVMTypes::Int64VectorType)
        funcName = g->target.is32Bit ? "__pseudo_gather32_64" : 
            "__pseudo_gather64_64";
    else if (llvmReturnType == LLVMTypes::FloatVectorType || 
             llvmReturnType == LLVMTypes::Int32VectorType)
        funcName = g->target.is32Bit ? "__pseudo_gather32_32" : 
            "__pseudo_gather64_32";
    else if (llvmReturnType == LLVMTypes::Int16VectorType)
        funcName = g->target.is32Bit ? "__pseudo_gather32_16" : 
            "__pseudo_gather64_16";
    else {
        Assert(llvmReturnType == LLVMTypes::Int8VectorType);
        funcName = g->target.is32Bit ? "__pseudo_gather32_8" : 
            "__pseudo_gather64_8";
    }

    llvm::Function *gatherFunc = m->module->getFunction(funcName);
    Assert(gatherFunc != NULL);

    llvm::Value *call = CallInst(gatherFunc, NULL, ptr, mask, name);

    // Add metadata about the source file location so that the
    // optimization passes can print useful performance warnings if we
    // can't optimize out this gather
    addGSMetadata(call, currentPos);

    return BitCastInst(call, llvmReturnType, "gather_bitcast");
}


/** Add metadata to the given instruction to encode the current source file
    position.  This data is used in the lGetSourcePosFromMetadata()
    function in opt.cpp. 
*/
void
FunctionEmitContext::addGSMetadata(llvm::Value *v, SourcePos pos) {
    llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(v);
    if (inst == NULL)
        return;

    llvm::Value *str = llvm::MDString::get(*g->ctx, pos.name);
    llvm::MDNode *md = llvm::MDNode::get(*g->ctx, str);
    inst->setMetadata("filename", md);

    llvm::Value *first_line = LLVMInt32(pos.first_line);
    md = llvm::MDNode::get(*g->ctx, first_line);
    inst->setMetadata("first_line", md);

    llvm::Value *first_column = LLVMInt32(pos.first_column);
    md = llvm::MDNode::get(*g->ctx, first_column);
    inst->setMetadata("first_column", md);

    llvm::Value *last_line = LLVMInt32(pos.last_line);
    md = llvm::MDNode::get(*g->ctx, last_line);
    inst->setMetadata("last_line", md);

    llvm::Value *last_column = LLVMInt32(pos.last_column);
    md = llvm::MDNode::get(*g->ctx, last_column);
    inst->setMetadata("last_column", md);
}


llvm::Value *
FunctionEmitContext::AllocaInst(LLVM_TYPE_CONST llvm::Type *llvmType, 
                                const char *name, int align, 
                                bool atEntryBlock) {
    if (llvmType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    llvm::AllocaInst *inst = NULL;
    if (atEntryBlock) {
        // We usually insert it right before the jump instruction at the
        // end of allocaBlock
        llvm::Instruction *retInst = allocaBlock->getTerminator();
        Assert(retInst);
        inst = new llvm::AllocaInst(llvmType, name ? name : "", retInst);
    }
    else
        // Unless the caller overrode the default and wants it in the
        // current basic block
        inst = new llvm::AllocaInst(llvmType, name ? name : "", bblock);

    // If no alignment was specified but we have an array of a uniform
    // type, then align it to 4 * the native vector width; it's not
    // unlikely that this array will be loaded into varying variables with
    // what will be aligned accesses if the uniform -> varying load is done
    // in regular chunks.
    LLVM_TYPE_CONST llvm::ArrayType *arrayType = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::ArrayType>(llvmType);
    if (align == 0 && arrayType != NULL && 
        !llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(arrayType->getElementType()))
        align = 4 * g->target.nativeVectorWidth;

    if (align != 0)
        inst->setAlignment(align);
    // Don't add debugging info to alloca instructions
    return inst;
}


/** Code to store the given varying value to the given location, only
    storing the elements that correspond to active program instances as
    given by the provided storeMask value.  Note that the lvalue is only a
    single pointer, not a varying lvalue of one pointer per program
    instance (that case is handled by scatters).
 */
void
FunctionEmitContext::maskedStore(llvm::Value *value, llvm::Value *ptr,
                                 const Type *ptrType, llvm::Value *mask) {
    if (value == NULL || ptr == NULL) {
        Assert(m->errorCount > 0);
        return;
    }

    Assert(dynamic_cast<const PointerType *>(ptrType) != NULL);
    Assert(ptrType->IsUniformType());

    const Type *valueType = ptrType->GetBaseType();
    const CollectionType *collectionType = 
        dynamic_cast<const CollectionType *>(valueType);
    if (collectionType != NULL) {
        // Assigning a structure / array / vector. Handle each element
        // individually with what turns into a recursive call to
        // makedStore()
        for (int i = 0; i < collectionType->GetElementCount(); ++i) {
            llvm::Value *eltValue = ExtractInst(value, i, "value_member");
            llvm::Value *eltPtr = 
                AddElementOffset(ptr, i, ptrType, "struct_ptr_ptr");
            const Type *eltPtrType = 
                PointerType::GetUniform(collectionType->GetElementType(i));
            StoreInst(eltValue, eltPtr, mask, eltPtrType);
        }
        return;
    }

    // We must have a regular atomic, enumerator, or pointer type at this
    // point.
    Assert(dynamic_cast<const AtomicType *>(valueType) != NULL ||
           dynamic_cast<const EnumType *>(valueType) != NULL ||
           dynamic_cast<const PointerType *>(valueType) != NULL);
    valueType = valueType->GetAsNonConstType();

    llvm::Function *maskedStoreFunc = NULL;
    // Figure out if we need a 8, 16, 32 or 64-bit masked store.
    if (dynamic_cast<const PointerType *>(valueType) != NULL) {
        if (g->target.is32Bit)
            maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_32");
        else
            maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_64");
    }
    else if (valueType == AtomicType::VaryingBool &&
             g->target.maskBitCount == 1) {
        llvm::Value *notMask = BinaryOperator(llvm::Instruction::Xor, mask,
                                              LLVMMaskAllOn, "~mask");
        llvm::Value *old = LoadInst(ptr);
        llvm::Value *maskedOld = BinaryOperator(llvm::Instruction::And, old,
                                                notMask, "old&~mask");
        llvm::Value *maskedNew = BinaryOperator(llvm::Instruction::And, value,
                                                mask, "new&mask");
        llvm::Value *final = BinaryOperator(llvm::Instruction::Or, maskedOld,
                                            maskedNew, "old_new_result");
        StoreInst(final, ptr);
        return;
    }
    else if (valueType == AtomicType::VaryingDouble || 
             valueType == AtomicType::VaryingInt64 ||
             valueType == AtomicType::VaryingUInt64) {
        maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_64");
        ptr = BitCastInst(ptr, LLVMTypes::Int64VectorPointerType, 
                             "ptr_to_int64vecptr");
        value = BitCastInst(value, LLVMTypes::Int64VectorType, 
                             "value_to_int64");
    }
    else if (valueType == AtomicType::VaryingFloat ||
             valueType == AtomicType::VaryingBool ||
             valueType == AtomicType::VaryingInt32 ||
             valueType == AtomicType::VaryingUInt32 ||
             dynamic_cast<const EnumType *>(valueType) != NULL) {
        maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_32");
        ptr = BitCastInst(ptr, LLVMTypes::Int32VectorPointerType, 
                             "ptr_to_int32vecptr");
        if (valueType == AtomicType::VaryingFloat)
            value = BitCastInst(value, LLVMTypes::Int32VectorType, 
                                 "value_to_int32");
    }
    else if (valueType == AtomicType::VaryingInt16 ||
             valueType == AtomicType::VaryingUInt16) {
        maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_16");
        ptr = BitCastInst(ptr, LLVMTypes::Int16VectorPointerType, 
                             "ptr_to_int16vecptr");
    }
    else if (valueType == AtomicType::VaryingInt8 ||
             valueType == AtomicType::VaryingUInt8) {
        maskedStoreFunc = m->module->getFunction("__pseudo_masked_store_8");
        ptr = BitCastInst(ptr, LLVMTypes::Int8VectorPointerType, 
                             "ptr_to_int8vecptr");
    }
    Assert(maskedStoreFunc != NULL);

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
void
FunctionEmitContext::scatter(llvm::Value *value, llvm::Value *ptr, 
                             const Type *ptrType, llvm::Value *mask) {
    Assert(dynamic_cast<const PointerType *>(ptrType) != NULL);
    Assert(ptrType->IsVaryingType());

    const Type *valueType = ptrType->GetBaseType();

    // I think this should be impossible
    Assert(dynamic_cast<const ArrayType *>(valueType) == NULL);

    const CollectionType *collectionType = dynamic_cast<const CollectionType *>(valueType);
    if (collectionType != NULL) {
        // Scatter the collection elements individually
        for (int i = 0; i < collectionType->GetElementCount(); ++i) {
            llvm::Value *eltPtr = AddElementOffset(ptr, i, ptrType);
            llvm::Value *eltValue = ExtractInst(value, i);
            const Type *eltPtrType = 
                PointerType::GetVarying(collectionType->GetElementType(i));
            eltPtr = addVaryingOffsetsIfNeeded(eltPtr, eltPtrType);
            scatter(eltValue, eltPtr, eltPtrType, mask);
        }
        return;
    }

    const PointerType *pt = dynamic_cast<const PointerType *>(valueType);

    // And everything should be a pointer or atomic from here on out...
    Assert(pt != NULL || 
           dynamic_cast<const AtomicType *>(valueType) != NULL);

    LLVM_TYPE_CONST llvm::Type *type = value->getType();
    const char *funcName = NULL;
    if (pt != NULL)
        funcName = g->target.is32Bit ? "__pseudo_scatter32_32" :
            "__pseudo_scatter64_64";
    else if (type == LLVMTypes::DoubleVectorType || 
             type == LLVMTypes::Int64VectorType) {
        funcName = g->target.is32Bit ? "__pseudo_scatter32_64" :
            "__pseudo_scatter64_64";
        value = BitCastInst(value, LLVMTypes::Int64VectorType, "value2int");
    }
    else if (type == LLVMTypes::FloatVectorType || 
             type == LLVMTypes::Int32VectorType) {
        funcName = g->target.is32Bit ? "__pseudo_scatter32_32" :
            "__pseudo_scatter64_32";
        value = BitCastInst(value, LLVMTypes::Int32VectorType, "value2int");
    }
    else if (type == LLVMTypes::Int16VectorType)
        funcName = g->target.is32Bit ? "__pseudo_scatter32_16" :
            "__pseudo_scatter64_16";
    else if (type == LLVMTypes::Int8VectorType)
        funcName = g->target.is32Bit ? "__pseudo_scatter32_8" :
            "__pseudo_scatter64_8";

    llvm::Function *scatterFunc = m->module->getFunction(funcName);
    Assert(scatterFunc != NULL);
    
    AddInstrumentationPoint("scatter");

    std::vector<llvm::Value *> args;
    args.push_back(ptr);
    args.push_back(value);
    args.push_back(mask);
    llvm::Value *inst = CallInst(scatterFunc, NULL, args);
    addGSMetadata(inst, currentPos);
}


void
FunctionEmitContext::StoreInst(llvm::Value *value, llvm::Value *ptr) {
    if (value == NULL || ptr == NULL) {
        // may happen due to error elsewhere
        Assert(m->errorCount > 0);
        return;
    }

    llvm::Instruction *inst;
    if (llvm::isa<llvm::VectorType>(value->getType()))
        // FIXME: same for load--do we still need/want this??
        // Specify an unaligned store, since we don't know that the ptr
        // will in fact be aligned to a vector width here.  (Actually
        // should be aligned to the alignment of the vector elment type...)
        inst = new llvm::StoreInst(value, ptr, false /* not volatile */,
                                   1, bblock);
    else
        inst = new llvm::StoreInst(value, ptr, bblock);

    AddDebugPos(inst);
}


void
FunctionEmitContext::StoreInst(llvm::Value *value, llvm::Value *ptr,
                               llvm::Value *mask, const Type *ptrType) {
    if (value == NULL || ptr == NULL) {
        // may happen due to error elsewhere
        Assert(m->errorCount > 0);
        return;
    }

    if (dynamic_cast<const ReferenceType *>(ptrType) != NULL)
        ptrType = PointerType::GetUniform(ptrType->GetReferenceTarget());

    // Figure out what kind of store we're doing here
    if (ptrType->IsUniformType()) {
        if (ptrType->GetBaseType()->IsUniformType())
            // the easy case
            StoreInst(value, ptr);
        else if (mask == LLVMMaskAllOn && !g->opt.disableMaskAllOnOptimizations)
            // Otherwise it is a masked store unless we can determine that the
            // mask is all on...  (Unclear if this check is actually useful.)
            StoreInst(value, ptr);
        else
            maskedStore(value, ptr, ptrType, mask);
    }
    else {
        Assert(ptrType->IsVaryingType());
        // We have a varying ptr (an array of pointers), so it's time to
        // scatter
        scatter(value, ptr, ptrType, GetFullMask());
    }
}


void
FunctionEmitContext::BranchInst(llvm::BasicBlock *dest) {
    llvm::Instruction *b = llvm::BranchInst::Create(dest, bblock);
    AddDebugPos(b);
}


void
FunctionEmitContext::BranchInst(llvm::BasicBlock *trueBlock, 
                                llvm::BasicBlock *falseBlock,
                                llvm::Value *test) {
    if (test == NULL) {
        Assert(m->errorCount > 0);
        return;
    }

    llvm::Instruction *b = 
        llvm::BranchInst::Create(trueBlock, falseBlock, test, bblock);
    AddDebugPos(b);
}


llvm::Value *
FunctionEmitContext::ExtractInst(llvm::Value *v, int elt, const char *name) {
    if (v == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    llvm::Instruction *ei = NULL;
    if (llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(v->getType()))
        ei = llvm::ExtractElementInst::Create(v, LLVMInt32(elt), 
                                              name ? name : "extract", bblock);
    else
        ei = llvm::ExtractValueInst::Create(v, elt, name ? name : "extract",
                                            bblock);
    AddDebugPos(ei);
    return ei;
}


llvm::Value *
FunctionEmitContext::InsertInst(llvm::Value *v, llvm::Value *eltVal, int elt, 
                                const char *name) {
    if (v == NULL || eltVal == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    llvm::Instruction *ii = NULL;
    if (llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(v->getType()))
        ii = llvm::InsertElementInst::Create(v, eltVal, LLVMInt32(elt), 
                                             name ? name : "insert", bblock);
    else
        ii = llvm::InsertValueInst::Create(v, eltVal, elt, 
                                           name ? name : "insert", bblock);
    AddDebugPos(ii);
    return ii;
}


llvm::PHINode *
FunctionEmitContext::PhiNode(LLVM_TYPE_CONST llvm::Type *type, int count, 
                             const char *name) {
    llvm::PHINode *pn = llvm::PHINode::Create(type, 
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
                                              count, 
#endif // LLVM_3_0
                                              name ? name : "phi", bblock);
    AddDebugPos(pn);
    return pn;
}


llvm::Instruction *
FunctionEmitContext::SelectInst(llvm::Value *test, llvm::Value *val0,
                                llvm::Value *val1, const char *name) {
    if (test == NULL || val0 == NULL || val1 == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    llvm::Instruction *inst = 
        llvm::SelectInst::Create(test, val0, val1, name ? name : "select", 
                                 bblock);
    AddDebugPos(inst);
    return inst;
}


/** Given a value representing a function to be called or possibly-varying
    pointer to a function to be called, figure out how many arguments the
    function has. */
static unsigned int
lCalleeArgCount(llvm::Value *callee, const FunctionType *funcType) {
    LLVM_TYPE_CONST llvm::FunctionType *ft = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::FunctionType>(callee->getType());

    if (ft == NULL) {
        LLVM_TYPE_CONST llvm::PointerType *pt =
            llvm::dyn_cast<LLVM_TYPE_CONST llvm::PointerType>(callee->getType());
        if (pt == NULL) {
            // varying--in this case, it must be the version of the
            // function that takes a mask
            return funcType->GetNumParameters() + 1;
        }
        ft = llvm::dyn_cast<LLVM_TYPE_CONST llvm::FunctionType>(pt->getElementType());
    }

    Assert(ft != NULL);
    return ft->getNumParams();
}


llvm::Value *
FunctionEmitContext::CallInst(llvm::Value *func, const FunctionType *funcType,
                              const std::vector<llvm::Value *> &args,
                              const char *name) {
    if (func == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    std::vector<llvm::Value *> argVals = args;
    // Most of the time, the mask is passed as the last argument.  this
    // isn't the case for things like intrinsics, builtins, and extern "C"
    // functions from the application.  Add the mask if it's needed.
    unsigned int calleeArgCount = lCalleeArgCount(func, funcType);
    Assert(argVals.size() + 1 == calleeArgCount ||
           argVals.size() == calleeArgCount);
    if (argVals.size() + 1 == calleeArgCount)
        argVals.push_back(GetFullMask());

    if (llvm::isa<LLVM_TYPE_CONST llvm::VectorType>(func->getType()) == false) {
        // Regular 'uniform' function call--just one function or function
        // pointer, so just emit the IR directly.
#if defined(LLVM_3_0) || defined(LLVM_3_0svn) || defined(LLVM_3_1svn)
        llvm::Instruction *ci = 
            llvm::CallInst::Create(func, argVals, name ? name : "", bblock);
#else
        llvm::Instruction *ci = 
            llvm::CallInst::Create(func, argVals.begin(), argVals.end(), 
                                   name ? name : "", bblock);
#endif
        AddDebugPos(ci);
        return ci;
    }
    else {
        // Emit the code for a varying function call, where we have an
        // vector of function pointers, one for each program instance.  The
        // basic strategy is that we go through the function pointers, and
        // for the executing program instances, for each unique function
        // pointer that's in the vector, call that function with a mask
        // equal to the set of active program instances that also have that
        // function pointer.  When all unique function pointers have been
        // called, we're done.

        llvm::BasicBlock *bbTest = CreateBasicBlock("varying_funcall_test");
        llvm::BasicBlock *bbCall = CreateBasicBlock("varying_funcall_call");
        llvm::BasicBlock *bbDone = CreateBasicBlock("varying_funcall_done");

        // Get the current mask value so we can restore it later
        llvm::Value *origMask = GetInternalMask();

        // First allocate memory to accumulate the various program
        // instances' return values...
        const Type *returnType = funcType->GetReturnType();
        LLVM_TYPE_CONST llvm::Type *llvmReturnType = returnType->LLVMType(g->ctx);
        llvm::Value *resultPtr = NULL;
        if (llvmReturnType->isVoidTy() == false)
            resultPtr = AllocaInst(llvmReturnType);

        // The memory pointed to by maskPointer tracks the set of program
        // instances for which we still need to call the function they are
        // pointing to.  It starts out initialized with the mask of
        // currently running program instances.
        llvm::Value *maskPtr = AllocaInst(LLVMTypes::MaskType);
        StoreInst(GetFullMask(), maskPtr);

        // And now we branch to the test to see if there's more work to be
        // done.
        BranchInst(bbTest);

        // bbTest: are any lanes of the mask still on?  If so, jump to
        // bbCall
        SetCurrentBasicBlock(bbTest); {
            llvm::Value *maskLoad = LoadInst(maskPtr);
            llvm::Value *any = Any(maskLoad);
            BranchInst(bbCall, bbDone, any);
        }

        // bbCall: this is the body of the loop that calls out to one of
        // the active function pointer values.
        SetCurrentBasicBlock(bbCall); {
            // Figure out the first lane that still needs its function
            // pointer to be called.
            llvm::Value *currentMask = LoadInst(maskPtr);
            llvm::Function *cttz = 
                m->module->getFunction("__count_trailing_zeros_i32");
            Assert(cttz != NULL);
            llvm::Value *firstLane = CallInst(cttz, NULL, LaneMask(currentMask),
                                              "first_lane");

            // Get the pointer to the function we're going to call this
            // time through: ftpr = func[firstLane]
            llvm::Value *fptr = 
                llvm::ExtractElementInst::Create(func, firstLane, 
                                                 "extract_fptr", bblock);

            // Smear it out into an array of function pointers
            llvm::Value *fptrSmear = SmearUniform(fptr, "func_ptr");

            // fpOverlap = (fpSmearAsVec == fpOrigAsVec).  This gives us a
            // mask for the set of program instances that have the same
            // value for their function pointer.
            llvm::Value *fpOverlap = 
                CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ,
                        fptrSmear, func);
            fpOverlap = I1VecToBoolVec(fpOverlap);

            // Figure out the mask to use when calling the function
            // pointer: we need to AND the current execution mask to handle
            // the case of any non-running program instances that happen to
            // have this function pointer value.
            // callMask = (currentMask & fpOverlap)
            llvm::Value *callMask = 
                BinaryOperator(llvm::Instruction::And, currentMask, fpOverlap,
                               "call_mask");

            // Set the mask
            SetInternalMask(callMask);

            // bitcast the i32/64 function pointer to the actual function
            // pointer type (the variant that includes a mask).
            LLVM_TYPE_CONST llvm::Type *llvmFuncType =
                funcType->LLVMFunctionType(g->ctx, true);
            LLVM_TYPE_CONST llvm::Type *llvmFPtrType = 
                llvm::PointerType::get(llvmFuncType, 0);
            llvm::Value *fptrCast = IntToPtrInst(fptr, llvmFPtrType);

            // Call the function: callResult = call ftpr(args, args, call mask)
            llvm::Value *callResult = CallInst(fptrCast, funcType, args, name);

            // Now, do a masked store into the memory allocated to
            // accumulate the result using the call mask.
            if (callResult != NULL) {
                Assert(resultPtr != NULL);
                StoreInst(callResult, resultPtr, callMask, 
                          PointerType::GetUniform(returnType));
            }
            else
                Assert(resultPtr == NULL);

            // Update the mask to turn off the program instances for which
            // we just called the function.
            // currentMask = currentMask & ~callmask
            llvm::Value *notCallMask = 
                BinaryOperator(llvm::Instruction::Xor, callMask, LLVMMaskAllOn,
                               "~callMask");
            currentMask = BinaryOperator(llvm::Instruction::And, currentMask, 
                                         notCallMask, "currentMask&~callMask");
            StoreInst(currentMask, maskPtr);

            // And go back to the test to see if we need to do another
            // call.
            BranchInst(bbTest);
        }

        // bbDone: We're all done; clean up and return the result we've
        // accumulated in the result memory.
        SetCurrentBasicBlock(bbDone);
        SetInternalMask(origMask);
        return resultPtr ? LoadInst(resultPtr) : NULL;
    }
}


llvm::Value *
FunctionEmitContext::CallInst(llvm::Value *func, const FunctionType *funcType,
                              llvm::Value *arg, const char *name) {
    std::vector<llvm::Value *> args;
    args.push_back(arg);
    return CallInst(func, funcType, args, name);
}


llvm::Value *
FunctionEmitContext::CallInst(llvm::Value *func, const FunctionType *funcType,
                              llvm::Value *arg0, llvm::Value *arg1, 
                              const char *name) {
    std::vector<llvm::Value *> args;
    args.push_back(arg0);
    args.push_back(arg1);
    return CallInst(func, funcType, args, name);
}


llvm::Instruction *
FunctionEmitContext::ReturnInst() {
    if (launchedTasks)
        // Add a sync call at the end of any function that launched tasks
        SyncInst();

    llvm::Instruction *rinst = NULL;
    if (returnValuePtr != NULL) {
        // We have value(s) to return; load them from their storage
        // location
        llvm::Value *retVal = LoadInst(returnValuePtr, "return_value");
        rinst = llvm::ReturnInst::Create(*g->ctx, retVal, bblock);
    }
    else {
        Assert(function->GetReturnType() == AtomicType::Void);
        rinst = llvm::ReturnInst::Create(*g->ctx, bblock);
    }

    AddDebugPos(rinst);
    bblock = NULL;
    return rinst;
}


llvm::Value *
FunctionEmitContext::LaunchInst(llvm::Value *callee, 
                                std::vector<llvm::Value *> &argVals,
                                llvm::Value *launchCount) {
    if (callee == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    launchedTasks = true;

    Assert(llvm::isa<llvm::Function>(callee));
    LLVM_TYPE_CONST llvm::Type *argType = 
        (llvm::dyn_cast<llvm::Function>(callee))->arg_begin()->getType();
    Assert(llvm::PointerType::classof(argType));
    LLVM_TYPE_CONST llvm::PointerType *pt = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::PointerType>(argType);
    Assert(llvm::StructType::classof(pt->getElementType()));
    LLVM_TYPE_CONST llvm::StructType *argStructType = 
        static_cast<LLVM_TYPE_CONST llvm::StructType *>(pt->getElementType());
    Assert(argStructType->getNumElements() == argVals.size() + 1);

    llvm::Function *falloc = m->module->getFunction("ISPCAlloc");
    Assert(falloc != NULL);
    llvm::Value *structSize = g->target.SizeOf(argStructType, bblock);
    if (structSize->getType() != LLVMTypes::Int64Type)
        // ISPCAlloc expects the size as an uint64_t, but on 32-bit
        // targets, SizeOf returns a 32-bit value
        structSize = ZExtInst(structSize, LLVMTypes::Int64Type,
                              "struct_size_to_64");
    int align = 4 * RoundUpPow2(g->target.nativeVectorWidth);

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

    // copy in the mask
    llvm::Value *mask = GetFullMask();
    llvm::Value *ptr = AddElementOffset(argmem, argVals.size(), NULL,
                                        "funarg_mask");
    StoreInst(mask, ptr);

    // And emit the call to the user-supplied task launch function, passing
    // a pointer to the task function being called and a pointer to the
    // argument block we just filled in
    llvm::Value *fptr = BitCastInst(callee, LLVMTypes::VoidPointerType);
    llvm::Function *flaunch = m->module->getFunction("ISPCLaunch");
    Assert(flaunch != NULL);
    std::vector<llvm::Value *> args;
    args.push_back(launchGroupHandlePtr);
    args.push_back(fptr);
    args.push_back(voidmem);
    args.push_back(launchCount);
    return CallInst(flaunch, NULL, args, "");
}


void
FunctionEmitContext::SyncInst() {
    llvm::Value *launchGroupHandle = LoadInst(launchGroupHandlePtr);
    llvm::Value *nullPtrValue = 
        llvm::Constant::getNullValue(LLVMTypes::VoidPointerType);
    llvm::Value *nonNull = CmpInst(llvm::Instruction::ICmp,
                                   llvm::CmpInst::ICMP_NE,
                                   launchGroupHandle, nullPtrValue);
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
llvm::Value *
FunctionEmitContext::addVaryingOffsetsIfNeeded(llvm::Value *ptr, 
                                               const Type *ptrType) {
    // This should only be called for varying pointers
    const PointerType *pt = dynamic_cast<const PointerType *>(ptrType);
    Assert(pt && pt->IsVaryingType());

    const Type *baseType = ptrType->GetBaseType();
    if (dynamic_cast<const AtomicType *>(baseType) == NULL &&
        dynamic_cast<const EnumType *>(baseType) == NULL &&
        dynamic_cast<const PointerType *>(baseType) == NULL)
        return ptr;
    if (baseType->IsUniformType())
        return ptr;
    
    // Find the size of a uniform element of the varying type
    LLVM_TYPE_CONST llvm::Type *llvmBaseUniformType = 
        baseType->GetAsUniformType()->LLVMType(g->ctx);
    llvm::Value *unifSize = g->target.SizeOf(llvmBaseUniformType, bblock);
    unifSize = SmearUniform(unifSize);

    // Compute offset = <0, 1, .. > * unifSize
    llvm::Value *varyingOffsets = llvm::UndefValue::get(unifSize->getType());
    for (int i = 0; i < g->target.vectorWidth; ++i) {
        llvm::Value *iValue = (g->target.is32Bit || g->opt.force32BitAddressing) ?
            LLVMInt32(i) : LLVMInt64(i);
        varyingOffsets = InsertInst(varyingOffsets, iValue, i, "varying_delta");
    }
    llvm::Value *offset = BinaryOperator(llvm::Instruction::Mul, unifSize, 
                                         varyingOffsets);
    
    if (g->opt.force32BitAddressing == true && g->target.is32Bit == false)
        // On 64-bit targets where we're doing 32-bit addressing
        // calculations, we need to convert to an i64 vector before adding
        // to the pointer
        offset = SExtInst(offset, LLVMTypes::Int64VectorType, "offset_to_64");

    return BinaryOperator(llvm::Instruction::Add, ptr, offset);
}


CFInfo *
FunctionEmitContext::popCFState() {
    Assert(controlFlowInfo.size() > 0);
    CFInfo *ci = controlFlowInfo.back();
    controlFlowInfo.pop_back();

    if (ci->IsSwitch()) {
        breakTarget = ci->savedBreakTarget;
        continueTarget = ci->savedContinueTarget;
        breakLanesPtr = ci->savedBreakLanesPtr;
        continueLanesPtr = ci->savedContinueLanesPtr;
        loopMask = ci->savedLoopMask;
        switchExpr = ci->savedSwitchExpr;
        defaultBlock = ci->savedDefaultBlock;
        caseBlocks = ci->savedCaseBlocks;
        nextBlocks = ci->savedNextBlocks;
        switchConditionWasUniform = ci->savedSwitchConditionWasUniform;
    }
    else if (ci->IsLoop() || ci->IsForeach()) {
        breakTarget = ci->savedBreakTarget;
        continueTarget = ci->savedContinueTarget;
        breakLanesPtr = ci->savedBreakLanesPtr;
        continueLanesPtr = ci->savedContinueLanesPtr;
        loopMask = ci->savedLoopMask;
    }
    else {
        Assert(ci->IsIf());
        // nothing to do
    }

    return ci;
}
