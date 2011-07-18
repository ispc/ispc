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

/** @file ctx.h
    @brief Declaration of the FunctionEmitContext class
*/

#ifndef ISPC_CTX_H
#define ISPC_CTX_H 1

#include "ispc.h"
#include <llvm/InstrTypes.h>
#include <llvm/Instructions.h>
#ifndef LLVM_2_8
#include <llvm/Analysis/DIBuilder.h>
#endif
#include <llvm/Analysis/DebugInfo.h>

struct CFInfo;

/** FunctionEmitContext is one of the key classes in ispc; it is used to
    help with emitting the intermediate representation of a function during
    compilation.  It carries information the current program context during
    IR emission (e.g. the basic block into which instructions should be
    added; or, the current source file and line number, so debugging
    symbols can be correctly generated).  This class also provides a number
    of helper routines that are useful for code that emits IR.
 */
class FunctionEmitContext {
public:
    /** Create a new FunctionEmitContext.
        @param returnType   The return type of the function
        @param function     LLVM function in the current module that corresponds
                            to the function
        @param funSym       Symbol that corresponds to the function
        @param firstStmtPos Source file position of the first statement in the
                            function
     */
    FunctionEmitContext(const Type *returnType, llvm::Function *function, Symbol *funSym,
                        SourcePos firstStmtPos);
    ~FunctionEmitContext();

    /** @name Current basic block management
        @{
     */
    /** Returns the current basic block pointer */ 
    llvm::BasicBlock *GetCurrentBasicBlock();
    
    /** Set the given llvm::BasicBlock to be the basic block to emit
        forthcoming instructions into. */
    void SetCurrentBasicBlock(llvm::BasicBlock *bblock);

    /** @name Mask management
        @{
     */
    /** Returns the current mask value */ 
    llvm::Value *GetMask();

    /** Provides the value of the mask at function entry */
    void SetEntryMask(llvm::Value *val);

    /** Sets the mask to a new value */
    void SetMask(llvm::Value *val);

    /** Sets the mask to (oldMask & val) */
    void MaskAnd(llvm::Value *oldMask, llvm::Value *val);

    /** Sets the mask to (oldMask & ~val) */
    void MaskAndNot(llvm::Value *oldMask, llvm::Value *test);

    /** Emits a branch instruction to the basic block btrue if any of the
        lanes of current mask are on and bfalse if none are on. */
    void BranchIfMaskAny(llvm::BasicBlock *btrue, llvm::BasicBlock *bfalse);

    /** Emits a branch instruction to the basic block btrue if all of the
        lanes of current mask are on and bfalse if none are on. */
    void BranchIfMaskAll(llvm::BasicBlock *btrue, llvm::BasicBlock *bfalse);

    /** Emits a branch instruction to the basic block btrue if none of the
        lanes of current mask are on and bfalse if none are on. */
    void BranchIfMaskNone(llvm::BasicBlock *btrue, llvm::BasicBlock *bfalse);
    /** @} */

    /** @name Control flow management
        @{
    */
    /** Notifies the FunctionEmitContext that we're starting emission of an
        'if' statement with a uniform test.  The value of the mask going
        into the 'if' statement is provided in the oldMask parameter. */
    void StartUniformIf(llvm::Value *oldMask);

    /** Notifies the FunctionEmitContext that we're starting emission of an
        'if' statement with a varying test.  The value of the mask going
        into the 'if' statement is provided in the oldMask parameter. */
    void StartVaryingIf(llvm::Value *oldMask);

    /** Notifies the FunctionEmitConitext that we're done emitting the IR
        for an 'if' statement. */
    void EndIf();

    /** Notifies the FunctionEmitContext that we're starting to emit IR
        for a loop.  Basic blocks are provides for where 'break' and
        'continue' statements should jump to (if all running lanes want to
        break or continue), uniformControlFlow indicates whether the loop
        condition is 'uniform', and oldMask provides the current mask going
        into the loop. */
    void StartLoop(llvm::BasicBlock *breakTarget, llvm::BasicBlock *continueTarget, 
                   bool uniformControlFlow, llvm::Value *oldMask);

    /** Informs FunctionEmitContext of the value of the mask at the start
        of a loop body. */
    void SetLoopMask(llvm::Value *mask);

    /** Informs FunctionEmitContext that code generation for a loop is
        finished. */
    void EndLoop();

    /** Emit code for a 'break' statement in a loop.  If doCoherenceCheck
        is true, then if we're in a 'varying' loop, code will be emitted to
        see if all of the lanes want to break, in which case a jump to the
        break target will be taken.  (For 'uniform' loops, the jump is
        always done). */
    void Break(bool doCoherenceCheck);

    /** Emit code for a 'continue' statement in a loop.  If
        doCoherenceCheck is true, then if we're in a 'varying' loop, code
        will be emitted to see if all of the lanes want to continue, in
        which case a jump to the continue target will be taken.  (For
        'uniform' loops, the jump is always done). */
    void Continue(bool doCoherenceCheck);

    /** This method is called by code emitting IR for a loop at the end of
        the loop body; it restores the lanes of the mask that executed a
        'continue' statement when going through the loop body in the
        previous iteration. */
    void RestoreContinuedLanes();

    /** Returns the current number of nested levels of 'varying' control
        flow */
    int VaryingCFDepth() const;

    /** Called to generate code for 'return' statement; value is the
        expression in the return statement (if non-NULL), and
        doCoherenceCheck indicates whether instructions should be generated
        to see if all of the currently-running lanes have returned (if
        we're under varying control flow).  */
    void CurrentLanesReturned(Expr *value, bool doCoherenceCheck);
    /** @} */

    /** @name Small helper/utility routines
        @{ 
    */
    /** Given a boolean mask value of type LLVMTypes::MaskType, return an
        i1 value that indicates if any of the mask lanes are on. */
    llvm::Value *Any(llvm::Value *mask);

    /** Given a boolean mask value of type LLVMTypes::MaskType, return an
        i1 value that indicates if all of the mask lanes are on. */
    llvm::Value *All(llvm::Value *mask);

    /** Given a boolean mask value of type LLVMTypes::MaskType, return an
        i32 value wherein the i'th bit is on if and only if the i'th lane
        of the mask is on. */
    llvm::Value *LaneMask(llvm::Value *mask);

    /** Given two masks of type LLVMTypes::MaskType, return an i1 value
        that indicates whether the two masks are equal. */
    llvm::Value *MasksAllEqual(llvm::Value *mask1, llvm::Value *mask2);

    /** Given a string, create an anonymous global variable to hold its
        value and return the pointer to the string. */
    llvm::Value *GetStringPtr(const std::string &str);

    /** Create a new basic block with given name */
    llvm::BasicBlock *CreateBasicBlock(const char *name);

    /** Given a vector with element type i1, return a vector of type
        LLVMTypes::BoolVectorType.  This method handles the conversion for
        the targets where the bool vector element type is, for example,
        i32. */
    llvm::Value *I1VecToBoolVec(llvm::Value *b);

    /** Emit code to call the user-supplied ISPCMalloc function to
        allocate space for an object of thee given type.  Returns the
        pointer value returned by the ISPCMalloc call. */
    llvm::Value *EmitMalloc(LLVM_TYPE_CONST llvm::Type *ty, int align = 0);

    /** Emit code to call the user-supplied ISPCFree function, passing it
        the given pointer to storage previously allocated by an
        EmitMalloc() call. */
    void EmitFree(llvm::Value *ptr);

    /** If the user has asked to compile the program with instrumentation,
        this inserts a callback to the user-supplied instrumentation
        function at the current point in the code. */
    void AddInstrumentationPoint(const char *note);
    /** @} */

    /** @name Debugging support
        @{
    */
    /** Set the current source file position; subsequent emitted
        instructions will have this position associated with them if
        debugging information is being generated. */
    void SetDebugPos(SourcePos pos);

    SourcePos GetDebugPos() const;

    /** Adds debugging metadata to the given instruction.  If pos == NULL,
        use FunctionEmitContext::currentPos as the source file position for
        the instruction.  Similarly, if a DIScope is provided, it's used
        and otherwise the scope is found from a GetDIScope() call.  This
        takes a llvm::Value for the instruction rather than an
        llvm::Instruction for convenience; in calling code we often have
        Instructions stored using Value pointers; the code here returns
        silently if it's not actually given an instruction. */
    void AddDebugPos(llvm::Value *instruction, const SourcePos *pos = NULL, 
                     llvm::DIScope *scope = NULL);

    /** Inform the debugging information generation code that a new scope
        is starting in the source program. */
    void StartScope();

    /** Inform the debugging information generation code that the current
        scope is ending in the source program. */
    void EndScope();

    /** Returns the llvm::DIScope corresponding to the current program
        scope. */
    llvm::DIScope GetDIScope() const;

    /** Emits debugging information for the variable represented by
        sym.  */
    void EmitVariableDebugInfo(Symbol *sym);

    /** Emits debugging information for the function parameter represented
        by sym.  */
    void EmitFunctionParameterDebugInfo(Symbol *sym);
    /** @} */

    /** @name IR instruction emission
        @brief These methods generally closely correspond to LLVM IR
        instructions.  See the LLVM assembly language reference manual
        (http://llvm.org/docs/LangRef.html) and the LLVM doxygen documentaion
        (http://llvm.org/doxygen) for more information.  Here we will only
        document significant generalizations to the functionality of the 
        corresponding basic LLVM instructions.

        Beyond actually emitting the instruction, the implementations of
        these methods in FunctionEmitContext also handle adding debugging
        metadata if debugging symbols are enabled, adding the instructions
        to the current basic block, and handling generalizations like
        'varying' lvalues, arithmetic operations with VectorType operands,
        etc.
        @{
    */
    /** Emit the binary operator given by the inst parameter.  If
        llvm::Values corresponding to VectorTypes are given as operands,
        this also handles applying the given operation to the vector
        elements. */
    llvm::Value *BinaryOperator(llvm::Instruction::BinaryOps inst,
                                llvm::Value *v0, llvm::Value *v1, 
                                const char *name = NULL);

    /** Emit the "not" operator.  Like BinaryOperator(), this also handles
        a VectorType-based operand. */
    llvm::Value *NotOperator(llvm::Value *v, const char *name = NULL);

    /** Emit a comparison instruction.  If the operands are VectorTypes,
        then a value for the corresponding boolean VectorType is
        returned. */
    llvm::Value *CmpInst(llvm::Instruction::OtherOps inst, 
                         llvm::CmpInst::Predicate pred,
                         llvm::Value *v0, llvm::Value *v1, const char *name = NULL);

    llvm::Value *BitCastInst(llvm::Value *value, LLVM_TYPE_CONST llvm::Type *type,
                             const char *name = NULL);
    llvm::Value *PtrToIntInst(llvm::Value *value, LLVM_TYPE_CONST llvm::Type *type,
                              const char *name = NULL);
    llvm::Value *IntToPtrInst(llvm::Value *value, LLVM_TYPE_CONST llvm::Type *type,
                              const char *name = NULL);
    llvm::Instruction *TruncInst(llvm::Value *value, LLVM_TYPE_CONST llvm::Type *type,
                                 const char *name = NULL);
    llvm::Instruction *CastInst(llvm::Instruction::CastOps op, llvm::Value *value,
                                LLVM_TYPE_CONST llvm::Type *type, const char *name = NULL);
    llvm::Instruction *FPCastInst(llvm::Value *value, LLVM_TYPE_CONST llvm::Type *type, 
                                  const char *name = NULL);
    llvm::Instruction *SExtInst(llvm::Value *value, LLVM_TYPE_CONST llvm::Type *type, 
                                const char *name = NULL);
    llvm::Instruction *ZExtInst(llvm::Value *value, LLVM_TYPE_CONST llvm::Type *type, 
                                const char *name = NULL);

    /** This GEP method is a generalization of the standard one in LLVM; it
        supports both uniform and varying basePtr values (an array of
        pointers) as well as uniform and varying index values (arrays of
        indices). */
    llvm::Value *GetElementPtrInst(llvm::Value *basePtr, llvm::Value *index0,
                                   llvm::Value *index1, const char *name = NULL);

    /** This is a convenience method to generate a GEP instruction with
        indices with values with known constant values as the ispc program
        is being compiled. */
    llvm::Value *GetElementPtrInst(llvm::Value *basePtr, int v0, int v1,
                                   const char *name = NULL);

    /** Load from the memory location(s) given by lvalue.  The lvalue may
        be varying, in which case this corresponds to a gather from the
        multiple memory locations given by the array of pointer values
        given by the lvalue.  If the lvalue is not varying, then the type
        parameter may be NULL. */
    llvm::Value *LoadInst(llvm::Value *lvalue, const Type *type,
                          const char *name = NULL);

    /** Emits an alloca instruction to allocate stack storage for the given
        type.  If a non-zero alignment is specified, the object is also
        allocated at the given alignment.  By default, the alloca
        instruction is added at the start of the function in the entry
        basic block; if it should be added to the current basic block, then
        the atEntryBlock parameter should be false. */ 
    llvm::Value *AllocaInst(LLVM_TYPE_CONST llvm::Type *llvmType, const char *name = NULL,
                            int align = 0, bool atEntryBlock = true);

    /** Standard store instruction; for this variant, the lvalue must be a
        single pointer, not a varying lvalue. */
    void StoreInst(llvm::Value *rvalue, llvm::Value *lvalue, 
                   const char *name = NULL);

    /** In this variant of StoreInst(), the lvalue may be varying.  If so,
        this corresponds to a scatter.  Whether the lvalue is uniform of
        varying, the given storeMask is used to mask the stores so that
        they only execute for the active program instances. */
    void StoreInst(llvm::Value *rvalue, llvm::Value *lvalue,
                   llvm::Value *storeMask, const Type *rvalueType,
                   const char *name = NULL);

    void BranchInst(llvm::BasicBlock *block);
    void BranchInst(llvm::BasicBlock *trueBlock, llvm::BasicBlock *falseBlock,
                    llvm::Value *test);

    /** This convenience method maps to an llvm::ExtractElementInst if the
        given value is a llvm::VectorType, and to an llvm::ExtractValueInst
        otherwise. */
    llvm::Value *ExtractInst(llvm::Value *v, int elt, const char *name = NULL);

    /** This convenience method maps to an llvm::InsertElementInst if the
        given value is a llvm::VectorType, and to an llvm::InsertValueInst
        otherwise. */
    llvm::Value *InsertInst(llvm::Value *v, llvm::Value *eltVal, int elt, 
                            const char *name = NULL);

    llvm::PHINode *PhiNode(LLVM_TYPE_CONST llvm::Type *type, int count, 
                           const char *name = NULL);
    llvm::Instruction *SelectInst(llvm::Value *test, llvm::Value *val0,
                                  llvm::Value *val1, const char *name = NULL);

    llvm::Instruction *CallInst(llvm::Function *func, 
                                const std::vector<llvm::Value *> &args,
                                const char *name = NULL);
    /** This is a convenience method that issues a call instruction to a
        function that takes just a single argument. */
    llvm::Instruction *CallInst(llvm::Function *func, llvm::Value *arg,
                                const char *name = NULL);

    /** This is a convenience method that issues a call instruction to a
        function that takes two arguments. */
    llvm::Instruction *CallInst(llvm::Function *func, llvm::Value *arg0,
                                llvm::Value *arg1, const char *name = NULL);

    /** Launch an asynchronous task to run the given function, passing it
        he given argument values. */
    llvm::Instruction *LaunchInst(llvm::Function *callee, 
                                  std::vector<llvm::Value *> &argVals);

    llvm::Instruction *ReturnInst();
    /** @} */

private:
    /** The basic block into which we add any alloca instructions that need
        to go at the very start of the function. */
    llvm::BasicBlock *allocaBlock;

    /** The current basic block into which we're emitting new
        instructions */
    llvm::BasicBlock *bblock;

    /** Pointer to stack-allocated memory that stores the current value of
        the program mask. */
    llvm::Value *maskPtr;

    /** Current source file position; if debugging information is being
        generated, this position is used to set file/line information for
        instructions. */
    SourcePos currentPos;

    /** Source file position where the function definition started.  Used
        for error messages and debugging symbols. */
    SourcePos funcStartPos;

    /** Type of result that the current function returns. */
    const Type *returnType;

    /** Value of the program mask when the function starts execution.  */
    llvm::Value *entryMask;

    /** If currently in a loop body, the value of the mask at the start of
        the loop. */
    llvm::Value *loopMask;

    /** If currently in a loop body, this is a pointer to memory to store a
        mask value that represents which of the lanes have executed a
        'break' statement.  If we're not in a loop body, this should be
        NULL. */
    llvm::Value *breakLanesPtr;

    /** Similar to breakLanesPtr, if we're inside a loop, this is a pointer
        to memory to record which of the program instances have executed a
        'continue' statement. */
    llvm::Value *continueLanesPtr;

    /** If we're inside a loop, this gives the basic block immediately
        after the current loop, which we will jump to if all of the lanes
        have executed a break statement or are otherwise done with the
        loop. */
    llvm::BasicBlock *breakTarget;

    /** If we're inside a loop, this gives the block to jump to if all of
        the running lanes have executed a 'continue' statement. */
    llvm::BasicBlock *continueTarget;

    /** A pointer to memory that records which of the program instances
        have executed a 'return' statement (and are thus really truly done
        running any more instructions in this functions. */
    llvm::Value *returnedLanesPtr;

    /** A pointer to memory to store the return value for the function.
        Since difference program instances may execute 'return' statements
        at different times, we need to accumulate the return values as they
        come in until we return for real. */
    llvm::Value *returnValuePtr;

    /** The CFInfo structure records information about a nesting level of
        control flow.  This vector lets us see what control flow is going
        around outside the current position in the function being
        emitted. */
    std::vector<CFInfo *> controlFlowInfo;

    /** DIFile object corresponding to the source file where the current
        function was defined (used for debugging info0. */
    llvm::DIFile diFile;

    /** DISubprogram corresponding to this function (used for debugging
        info). */
    llvm::DISubprogram diFunction;

    /** These correspond to the current set of nested scopes in the
        function. */
    std::vector<llvm::DILexicalBlock> debugScopes;

    /** True if a 'launch' statement has been encountered in the function. */
    bool launchedTasks;

    llvm::Value *pointerVectorToVoidPointers(llvm::Value *value);
    static void addGSMetadata(llvm::Instruction *inst, SourcePos pos);
    bool ifsInLoopAllUniform() const;
    void jumpIfAllLoopLanesAreDone(llvm::BasicBlock *target);
    llvm::Value *emitGatherCallback(llvm::Value *lvalue, llvm::Value *retPtr);

    void restoreMaskGivenReturns(llvm::Value *oldMask);

    void scatter(llvm::Value *rvalue, llvm::Value *lvalue, 
                 llvm::Value *maskPtr, const Type *rvalueType);
    llvm::Value *gather(llvm::Value *lvalue, const Type *type,
                        const char *name);
    void maskedStore(llvm::Value *rvalue, llvm::Value *lvalue,
                     const Type *rvalueType, llvm::Value *maskPtr);
};

#endif // ISPC_CTX_H
