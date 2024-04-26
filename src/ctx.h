/*
  Copyright (c) 2010-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file ctx.h
    @brief %Declaration of the FunctionEmitContext class
*/

#pragma once

#include "ispc.h"

#include <map>

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>

namespace ispc {

struct CFInfo;

///////////////////////////////////////////////////////////////////////////
/** AddressInfo is a helper class to work with pointers.
    It keeps llvm pointer, llvm element type, and ISPC type.
*/
class AddressInfo : public Traceable {
  public:
    AddressInfo(llvm::Value *p, llvm::Type *t);
    AddressInfo(llvm::Value *p, const Type *t);
    llvm::Value *getPointer() const { return pointer; }

    // Return the type of the pointer value.
    llvm::PointerType *getType() const { return llvm::cast<llvm::PointerType>(getPointer()->getType()); }

    // Return the type of the values stored in this address.
    llvm::Type *getElementType() const { return elementType; }

    // Return the ISPC type. May be nullptr.
    const Type *getISPCType() const { return ispcType; }

    // Return the address space that this address resides in.
    unsigned getAddressSpace() const { return getType()->getAddressSpace(); }

    // Get LLVM pointer element type from ISPC PointerType.
    static llvm::Type *GetPointeeLLVMType(const PointerType *pt);

  private:
    llvm::Value *pointer;
    llvm::Type *elementType;
    const Type *ispcType;
};

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
        @param function     The Function object representing the function
        @param funSym       Symbol that corresponds to the function
        @param llvmFunction LLVM function in the current module that corresponds
                            to the function
        @param firstStmtPos Source file position of the first statement in the
                            function
     */
    FunctionEmitContext(Function *function, Symbol *funSym, llvm::Function *llvmFunction, SourcePos firstStmtPos);
    ~FunctionEmitContext();

    /** Returns the Function * corresponding to the function that we're
        currently generating code for. */
    const Function *GetFunction() const;

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
    /** Returns the mask value at entry to the current function. */
    llvm::Value *GetFunctionMask();

    /** Returns the mask value corresponding to "varying" control flow
        within the current function.  (i.e. this doesn't include the effect
        of the mask at function entry. */
    llvm::Value *GetInternalMask();

    /** Returns the complete current mask value--i.e. the logical AND of
        the function entry mask and the internal mask. */
    llvm::Value *GetFullMask();

    /** Returns an AddressInfo with a pointer to storage in memory that stores the current full
        mask. */
    AddressInfo *GetFullMaskAddressInfo();

    /** Provides the value of the mask at function entry */
    void SetFunctionMask(llvm::Value *val);

    /** Sets the internal mask to a new value */
    void SetInternalMask(llvm::Value *val);

    /** Sets the internal mask to (oldMask & val) */
    void SetInternalMaskAnd(llvm::Value *oldMask, llvm::Value *val);

    /** Sets the internal mask to (oldMask & ~val) */
    void SetInternalMaskAndNot(llvm::Value *oldMask, llvm::Value *test);

    /** Emits a branch instruction to the basic block btrue if any of the
        lanes of current mask are on and bfalse if none are on. */
    llvm::Instruction *BranchIfMaskAny(llvm::BasicBlock *btrue, llvm::BasicBlock *bfalse);

    /** Emits a branch instruction to the basic block btrue if all of the
        lanes of current mask are on and bfalse if none are on. */
    void BranchIfMaskAll(llvm::BasicBlock *btrue, llvm::BasicBlock *bfalse);

    /** @name Control flow management
        @{
    */
    /** Notifies the FunctionEmitContext that we're starting emission of an
        'if' statement with a uniform test. If we emulate uniform branch as
        for GEN, emulateUniform must be true
        */
    void StartUniformIf(bool emulateUniform = false);

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
        condition is 'uniform'. */
    void StartLoop(llvm::BasicBlock *breakTarget, llvm::BasicBlock *continueTarget, bool uniformControlFlow,
                   bool isEmulatedUniform = false);

    /** Informs FunctionEmitContext of the value of the mask at the start
        of a loop body or switch statement. */
    void SetBlockEntryMask(llvm::Value *mask);

    /** Informs FunctionEmitContext that code generation for a loop is
        finished. */
    void EndLoop();

    /** Indicates that code generation for a 'foreach', 'foreach_tiled',
        'foreach_active', or 'foreach_unique' loop is about to start. */
    enum ForeachType { FOREACH_REGULAR, FOREACH_ACTIVE, FOREACH_UNIQUE };
    void StartForeach(ForeachType ft, bool isEmulatedUniform = false);
    void EndForeach();

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

    /** This method is called by code emitting IR for a loop.  It clears
        any lanes that contained a break since the mask has been updated to take
        them into account.  This is necessary as all the bail out checks for
        breaks are meant to only deal with lanes breaking on the current iteration.
     */
    void ClearBreakLanes();

    /** Indicates that code generation for a "switch" statement is about to
        start.  isUniform indicates whether the "switch" value is uniform,
        and bbAfterSwitch gives the basic block immediately following the
        "switch" statement.  (For example, if the switch condition is
        uniform, we jump here upon executing a "break" statement.) */
    void StartSwitch(bool isUniform, llvm::BasicBlock *bbAfterSwitch, bool isEmulatedUniform = false);
    /** Indicates the end of code generation for a "switch" statement. */
    void EndSwitch();

    /** Emits code for a "switch" statement in the program.
        @param expr         Gives the value of the expression after the "switch"
        @param defaultBlock Basic block to execute for the "default" case.  This
                            should be nullptr if there is no "default" label inside
                            the switch.
        @param caseBlocks   vector that stores the mapping from label values
                            after "case" statements to basic blocks corresponding
                            to the "case" labels.
        @param nextBlocks   For each basic block for a "case" or "default"
                            label, this gives the basic block for the
                            immediately-following "case" or "default" label (or
                            the basic block after the "switch" statement for the
                            last label.)
    */
    void SwitchInst(llvm::Value *expr, llvm::BasicBlock *defaultBlock,
                    const std::vector<std::pair<int, llvm::BasicBlock *>> &caseBlocks,
                    const std::map<llvm::BasicBlock *, llvm::BasicBlock *> &nextBlocks);

    /** Generates code for a "default" label after a "switch" statement.
        The checkMask parameter indicates whether additional code should be
        generated to check to see if the execution mask is all off after
        the default label (in which case a jump to the following label will
        be issued. */
    void EmitDefaultLabel(bool checkMask, SourcePos pos);

    /** Generates code for a "case" label after a "switch" statement.  See
        the documentation for EmitDefaultLabel() for discussion of the
        checkMask parameter. */
    void EmitCaseLabel(int value, bool checkMask, SourcePos pos);

    /** Returns the current number of nested levels of 'varying' control
        flow */
    int VaryingCFDepth() const;

    bool InForeachLoop() const;

    /** Temporarily disables emission of performance warnings from gathers
        and scatters from subsequent code. */
    void DisableGatherScatterWarnings();

    /** Reenables emission of gather/scatter performance warnings. */
    void EnableGatherScatterWarnings();

    void SetContinueTarget(llvm::BasicBlock *bb) { continueTarget = bb; }

    /** Step through the code and find label statements; create a basic
        block for each one, so that subsequent calls to
        GetLabeledBasicBlock() return the corresponding basic block. */
    void InitializeLabelMap(Stmt *code);

    /** If there is a label in the function with the given name, return the
        new basic block that it starts. */
    llvm::BasicBlock *GetLabeledBasicBlock(const std::string &label);

    /** Returns a vector of all labels in the context. This is
        simply the key set of the labelMap */
    std::vector<std::string> GetLabels();

    /** Called to generate code for 'return' statement; value is the
        expression in the return statement (if non-nullptr), and
        doCoherenceCheck indicates whether instructions should be generated
        to see if all of the currently-running lanes have returned (if
        we're under varying control flow).  */
    void CurrentLanesReturned(Expr *value, bool doCoherenceCheck);
    /** @} */

    /** @name FTZ/DAZ-flags related routines
        @{
    */
    /** Sets FTZ/DAZ flags. Returns the value detected on function entry */
    void SetFunctionFTZ_DAZFlags();

    /** Restores FTZ/DAZ flags saved on function entry */
    void RestoreFunctionFTZ_DAZFlags();

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
        i1 value that indicates if all of the mask lanes are off. */
    llvm::Value *None(llvm::Value *mask);

    /** Given a boolean mask value of type LLVMTypes::MaskType, return an
        i64 value wherein the i'th bit is on if and only if the i'th lane
        of the mask is on. */
    llvm::Value *LaneMask(llvm::Value *mask);

    /** Given two masks of type LLVMTypes::MaskType, return an i1 value
        that indicates whether the two masks are equal. */
    llvm::Value *MasksAllEqual(llvm::Value *mask1, llvm::Value *mask2);

    /** generate constantvector, which contains programindex, i.e.
        < i32 0, i32 1, i32 2, i32 3> */
    llvm::Value *ProgramIndexVector(bool is32bits = true);

    /** Given a string, create an anonymous global variable to hold its
        value and return the pointer to the string. */
    llvm::Value *GetStringPtr(const std::string &str);

    /** Create a new basic block with given name */
    llvm::BasicBlock *CreateBasicBlock(const llvm::Twine &name, llvm::BasicBlock *insertAfter = nullptr);

    /** Given a vector with element type i1, return a vector of type
        LLVMTypes::BoolVectorType.  This method handles the conversion for
        the targets where the bool vector element type is, for example,
        i32. */
    llvm::Value *I1VecToBoolVec(llvm::Value *b);

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

    /** Adds debugging metadata to the given instruction.  If pos == nullptr,
        use FunctionEmitContext::currentPos as the source file position for
        the instruction.  Similarly, if a DIScope is provided, it's used
        and otherwise the scope is found from a GetDIScope() call.  This
        takes a llvm::Value for the instruction rather than an
        llvm::Instruction for convenience; in calling code we often have
        Instructions stored using Value pointers; the code here returns
        silently if it's not actually given an instruction. */
    void AddDebugPos(llvm::Value *instruction, const SourcePos *pos = nullptr, llvm::DIScope *scope = nullptr);
    // llvm::MDScope *scope = nullptr );

    /** Inform the debugging information generation code that a new scope
        is starting in the source program. */
    void StartScope();

    /** Inform the debugging information generation code that the current
        scope is ending in the source program. */
    void EndScope();

    /** Returns the llvm::DIScope corresponding to the current program
        scope. */

    llvm::DIScope *GetDIScope() const;

    /** Emits debugging information for the variable represented by
        sym.  */
    void EmitVariableDebugInfo(Symbol *sym);

    /** Emits debugging information for the function parameter represented
        by sym.  */
    void EmitFunctionParameterDebugInfo(Symbol *sym, int parameterNum);
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
        elements.

        The isSigned parameter toggles whether the nsw attribute is applied
        to signed integer arithmetic operations.
        The value of isSigned is determined either by
        1.  the sign of the incoming generated expression's type
            (e.g. using type->IsSignedType()), or
        2.  the sign of a constructed expression (e.g. false for pointer
            arithmetic, true for foreach induction variables).
        The only arithmetic oparations modified by this parameter are Add, Sub, and Mul.
        Shl supports the attribute (https://llvm.org/docs/LangRef.html#shl-instruction),
        but we elect not to emit nsw for these operations to match Clang/GCC behavior.
        Care should be used, as some optimizations may rely on undefined behavior
        for signed integer overflow.
        See the Expressions section of docs/ispc.rst for more information on this
        optimization.
    */
    llvm::Value *BinaryOperator(llvm::Instruction::BinaryOps inst, llvm::Value *v0, llvm::Value *v1,
                                WrapSemantics wrapSemantics, const llvm::Twine &name = "");

    /** Emit the "not" operator.  Like BinaryOperator(), this also handles
        a VectorType-based operand. */
    llvm::Value *NotOperator(llvm::Value *v, const llvm::Twine &name = "");

    /** Emit FNeg instruction. */
    llvm::Value *FNegInst(llvm::Value *v, const llvm::Twine &name = "");

    /** Emit a comparison instruction.  If the operands are VectorTypes,
        then a value for the corresponding boolean VectorType is
        returned. */
    llvm::Value *CmpInst(llvm::Instruction::OtherOps inst, llvm::CmpInst::Predicate pred, llvm::Value *v0,
                         llvm::Value *v1, const llvm::Twine &name = "");

    /** Given a scalar value, return a vector of the same type (or an
        array, for pointer types). */
    llvm::Value *SmearUniform(llvm::Value *value, const llvm::Twine &name = "");

    llvm::Value *BitCastInst(llvm::Value *value, llvm::Type *type, const llvm::Twine &name = "");
    llvm::Value *PtrToIntInst(llvm::Value *value, const llvm::Twine &name = "");
    llvm::Value *PtrToIntInst(llvm::Value *value, llvm::Type *type, const llvm::Twine &name = "");
    llvm::Value *IntToPtrInst(llvm::Value *value, llvm::Type *type, const llvm::Twine &name = "");

    llvm::Instruction *TruncInst(llvm::Value *value, llvm::Type *type, const llvm::Twine &name = "");
    llvm::Instruction *CastInst(llvm::Instruction::CastOps op, llvm::Value *value, llvm::Type *type,
                                const llvm::Twine &name = "");
    llvm::Instruction *FPCastInst(llvm::Value *value, llvm::Type *type, const llvm::Twine &name = "");
    llvm::Instruction *SExtInst(llvm::Value *value, llvm::Type *type, const llvm::Twine &name = "");
    llvm::Instruction *ZExtInst(llvm::Value *value, llvm::Type *type, const llvm::Twine &name = "");

    /** Given two integer-typed values (but possibly one vector and the
        other not, and or of possibly-different bit-widths), update their
        values as needed so that the two have the same (more general)
        type. */
    void MatchIntegerTypes(llvm::Value **v0, llvm::Value **v1);

    /** Create a new slice pointer out of the given pointer to an soa type
        and an integer offset to a slice within that type. */
    llvm::Value *MakeSlicePointer(llvm::Value *ptr, llvm::Value *offset);

    /* Regularize to a standard pointer type.
       May return nullptr if type is not PointerType or ReferenceType */
    const PointerType *RegularizePointer(const Type *ptrRefType);

    /** These GEP methods are generalizations of the standard ones in LLVM;
        they support both uniform and varying basePtr values as well as
        uniform and varying index values (arrays of indices).  Varying base
        pointers are expected to come in as vectors of i32/i64 (depending
        on the target), since LLVM doesn't currently support vectors of
        pointers.  The underlying type of the base pointer must be provided
        via the ptrType parameter */
    llvm::Value *GetElementPtrInst(llvm::Value *basePtr, llvm::Value *index, const Type *ptrType,
                                   const llvm::Twine &name = "");
    llvm::Value *GetElementPtrInst(llvm::Value *basePtr, llvm::Value *index0, llvm::Value *index1, const Type *ptrType,
                                   const llvm::Twine &name = "");

    /** This method returns a new pointer that represents offsetting the
        given base pointer to point at the given element number of the
        structure type that the base pointer points to.  (The provided
        pointer in AddressInfo must be a pointer to a structure type.) */
    llvm::Value *AddElementOffset(AddressInfo *basePtrInfo, int elementNum, const llvm::Twine &name = "",
                                  const PointerType **resultPtrType = nullptr);

    /** Bool is stored as i8 and <WIDTH x i8> (storage type) but represented in
     * IR as i1 and <WIDTH x MASK> (mask type). These are two helper functions
     * to match bool sizes.  */
    llvm::Value *SwitchBoolToMaskType(llvm::Value *value, llvm::Type *toType, const llvm::Twine &name = "");
    llvm::Value *SwitchBoolToStorageType(llvm::Value *value, llvm::Type *toType, const llvm::Twine &name = "");

    /** Load from the memory location(s) given by lvalue, using the given
        mask.  The lvalue may be varying, in which case this corresponds to
        a gather from the multiple memory locations given by the array of
        pointer values given by the lvalue.  If the lvalue is not varying,
        then both the mask pointer and the type pointer may be nullptr. */
    llvm::Value *LoadInst(llvm::Value *ptr, llvm::Value *mask, const Type *ptrType, const llvm::Twine &name = "",
                          bool one_elem = false);

    /* Load from memory location(s) given.
     * 'type' needs to be provided when storage type is different from IR type. For example,
     * 'unform bool' is 'i1' in IR but stored as 'i8'.
     * Otherwise leave this as nullptr. */
    llvm::Value *LoadInst(AddressInfo *ptrInfo, const Type *type = nullptr, const llvm::Twine &name = "");

    /** Emits addrspacecast instruction. Depending on atEntryBlock it is generated in
        alloca block or in the current block.
    */
    llvm::Value *AddrSpaceCastInst(llvm::Value *val, AddressSpace as, bool atEntryBlock = false);

    /** Emits an alloca instruction to allocate stack storage of the given
        size.  If a non-zero alignment is specified, the object is also
        allocated at the given alignment.  By default, the alloca
        instruction is added at the start of the function in the entry
        basic block; if it should be added to the current basic block, then
        the atEntryBlock parameter should be false. */
    AddressInfo *AllocaInst(llvm::Type *llvmType, llvm::Value *size, const llvm::Twine &name = "", int align = 0,
                            bool atEntryBlock = true);
    /** Emits an alloca instruction to allocate stack storage for the given
        type.  If a non-zero alignment is specified, the object is also
        allocated at the given alignment.  By default, the alloca
        instruction is added at the start of the function in the entry
        basic block; if it should be added to the current basic block, then
        the atEntryBlock parameter should be false. */
    AddressInfo *AllocaInst(llvm::Type *llvmType, const llvm::Twine &name = "", int align = 0,
                            bool atEntryBlock = true);

    /** Emits an alloca instruction to allocate stack storage for the given
        type.  If a non-zero alignment is specified, the object is also
        allocated at the given alignment.  By default, the alloca
        instruction is added at the start of the function in the entry
        basic block; if it should be added to the current basic block, then
        the atEntryBlock parameter should be false.
        This implementation is preferred when possible. It is needed when
        storage type is different from IR type. For example,
        'unform bool' is 'i1' in IR but stored as 'i8'. */
    AddressInfo *AllocaInst(const Type *ptrType, const llvm::Twine &name = "", int align = 0, bool atEntryBlock = true);

    /** Standard store instruction; for this variant, the lvalue must be a
        single pointer, not a varying lvalue.
        'ptrType' needs to be provided when storage type is different from IR type. For example,
        'unform bool' is 'i1' in IR but stored as 'i8'. */
    /*  TODO: keep all info about type in ptrInfo so we can eliminate usage of ptrType optional arg */
    void StoreInst(llvm::Value *value, AddressInfo *ptrInfo, const Type *ptrType = nullptr);

    /** In this variant of StoreInst(), the lvalue may be varying.  If so,
        this corresponds to a scatter.  Whether the lvalue is uniform of
        varying, the given storeMask is used to mask the stores so that
        they only execute for the active program instances. */
    void StoreInst(llvm::Value *value, llvm::Value *ptr, llvm::Value *storeMask, const Type *valueType,
                   const Type *ptrType);

    /** Copy count bytes of memory from the location pointed to by src to
        the location pointed to by dest.  (src and dest must not be
        overlapping.) */
    void MemcpyInst(llvm::Value *dest, llvm::Value *src, llvm::Value *count, llvm::Value *align = nullptr);

    void setLoopUnrollMetadata(llvm::Instruction *inst, std::pair<Globals::pragmaUnrollType, int> loopAttribute,
                               SourcePos pos);
    llvm::Instruction *BranchInst(llvm::BasicBlock *block);
    llvm::Instruction *BranchInst(llvm::BasicBlock *trueBlock, llvm::BasicBlock *falseBlock, llvm::Value *test);

    /** This convenience method maps to an llvm::ExtractElementInst if the
        given value is a llvm::VectorType, and to an llvm::ExtractValueInst
        otherwise. */
    llvm::Value *ExtractInst(llvm::Value *v, int elt, const llvm::Twine &name = "");

    /** This convenience method maps to an llvm::InsertElementInst if the
        given value is a llvm::VectorType, and to an llvm::InsertValueInst
        otherwise. */
    llvm::Value *InsertInst(llvm::Value *v, llvm::Value *eltVal, int elt, const llvm::Twine &name = "");

    /** This convenience method maps to an llvm::ShuffleVectorInst. */
    llvm::Value *ShuffleInst(llvm::Value *v1, llvm::Value *v2, llvm::Value *mask, const llvm::Twine &name = "");

    /** This convenience method to generate broadcast pattern. It takes a value
        and a vector type. Type of the value must match element type of the
        vector. */
    llvm::Value *BroadcastValue(llvm::Value *v, llvm::Type *vecType, const llvm::Twine &name = "");

    llvm::PHINode *PhiNode(llvm::Type *type, int count, const llvm::Twine &name = "");
    llvm::Instruction *SelectInst(llvm::Value *test, llvm::Value *val0, llvm::Value *val1,
                                  const llvm::Twine &name = "");

    /** Emits IR to do a function call with the given arguments.  If the
        function type is a varying function pointer type, its full type
        must be provided in funcType.  funcType can be nullptr if func is a
        uniform function pointer. */
    llvm::Value *CallInst(llvm::Value *func, const FunctionType *funcType, const std::vector<llvm::Value *> &args,
                          const llvm::Twine &name = "");

    /** This is a convenience method that issues a call instruction to a
        function that takes just a single argument. */
    llvm::Value *CallInst(llvm::Value *func, const FunctionType *funcType, llvm::Value *arg,
                          const llvm::Twine &name = "");

    /** This is a convenience method that issues a call instruction to a
        function that takes two arguments. */
    llvm::Value *CallInst(llvm::Value *func, const FunctionType *funcType, llvm::Value *arg0, llvm::Value *arg1,
                          const llvm::Twine &name = "");

    /** Launch an asynchronous task to run the given function, passing it
        he given argument values. */
    llvm::Value *LaunchInst(llvm::Value *callee, std::vector<llvm::Value *> &argVals, llvm::Value *launchCount[3],
                            const FunctionType *funcType);

    void SyncInst();

    llvm::Instruction *ReturnInst();

    /** Emits code for invoke_sycl*/
    llvm::Value *InvokeSyclInst(llvm::Value *func, const FunctionType *funcType,
                                const std::vector<llvm::Value *> &args);

#ifdef ISPC_XE_ENABLED
    /** Emit genx_simdcf_any intrinsic.
        Required when Xe hardware mask is emitted. */
    llvm::Value *XeSimdCFAny(llvm::Value *value);

    /** Emit genx_simdcf_predicate intrinsic
        Required when Xe hardware mask is emitted. */
    llvm::Value *XeSimdCFPredicate(llvm::Value *values, llvm::Value *defaults = nullptr);

    /** Start unmasked region. Sets execution mask to all-active, and return the old mask.*/
    llvm::Value *XeStartUnmaskedRegion();

    /** End unmasked region. Sets execution mask back using the value from unmask-begin.
        ISPCSIMDCFLowering expect that execMask have alloca+load+store. */
    void XeEndUnmaskedRegion(llvm::Value *execMask);

    /** Emit a string in constant space and get pointer to its first element.
        GEP constexpr is returned.
        Args:
          \p str - constant initializer;
          \p name - constant name */
    llvm::Constant *XeCreateConstantString(llvm::StringRef str, llvm::StringRef name = "");
    /** Similar to XeCreateConstantString but searches for the constant with the provided name first.
        If there's such constant returns pointer to its first element, otherwise creates new constant
        and returns pointer to its first element. */
    llvm::Constant *XeGetOrCreateConstantString(llvm::StringRef str, llvm::StringRef name);

    /** Change scalar condition to vector condition before branching if
        emulated uniform condition was found in external scopes and start SIMD control
        flow with simdcf.any intrinsic.
        Required when Xe hardware mask is emitted. */
    llvm::Value *XePrepareVectorBranch(llvm::Value *value);

    /** Emit ISPC-Uniform metadata to llvm instruction. Instruction with
        such metadata will not be predicated in ISPCSIMDCFLowering pass.
        Required when Xe hardware mask is emitted. */
    void XeUniformMetadata(llvm::Value *v);

    /** Check if current control flow block is inside Xe SIND CF
        Required when Xe hardware mask is emitted. */
    bool inXeSimdCF() const;

    /** This function checks addrspace of function parameter on paramIndex and returns
        val with casted addrspace if required. If cast is not required, original val is returned*/
    llvm::Value *XeUpdateAddrSpaceForParam(llvm::Value *val, const llvm::FunctionType *fType,
                                           const unsigned int paramIndex, bool atEntryBlock = false);

#endif
    /** Enables emitting of genx.any intrinsics and the control flow which is
        based on impliit hardware mask. Forces generation of goto/join instructions
        in assembly. */
    bool emitXeHardwareMask();

    /** @} */

  private:
    /** Pointer to the Function for which we're currently generating code. */
    Function *function;

    /** LLVM function representation for the current function. */
    llvm::Function *llvmFunction;

    /** The basic block into which we add any alloca instructions that need
        to go at the very start of the function. */
    llvm::BasicBlock *allocaBlock;

    /** The current basic block into which we're emitting new
        instructions */
    llvm::BasicBlock *bblock;

    /** AddressInfo with pointer to stack-allocated memory that stores the current value of
        the full program mask. */
    AddressInfo *fullMaskAddressInfo;

    /** AddressInfo with pointer to stack-allocated memory that stores the current value of
        the program mask representing varying control flow within the
        function. */
    AddressInfo *internalMaskAddressInfo;

    /** Value of the program mask when the function starts execution.  */
    llvm::Value *functionMaskValue;

    /** Value of the ftz/daz flags when the function starts execution.  */
    AddressInfo *functionFTZ_DAZValue;

    /** Current source file position; if debugging information is being
        generated, this position is used to set file/line information for
        instructions. */
    SourcePos currentPos;

    /** Source file position where the function definition started.  Used
        for error messages and debugging symbols. */
    SourcePos funcStartPos;

    /** If currently in a loop body or switch statement, the value of the
        mask at the start of it. */
    llvm::Value *blockEntryMask;

    /** If currently in a loop body or switch statement, this is an AddressInfo with pointer
        to memory to store a mask value that represents which of the lanes
        have executed a 'break' statement.  If we're not in a loop body or
        switch, this should be nullptr. */
    AddressInfo *breakLanesAddressInfo;

    /** Similar to breakLanesAddressInfo, if we're inside a loop, this is an AddressInfo with a pointer
        to memory to record which of the program instances have executed a
        'continue' statement. */
    AddressInfo *continueLanesAddressInfo;

    /** If we're inside a loop or switch statement, this gives the basic
        block immediately after the current loop or switch, which we will
        jump to if all of the lanes have executed a break statement or are
        otherwise done with it. */
    llvm::BasicBlock *breakTarget;

    /** If we're inside a loop, this gives the block to jump to if all of
        the running lanes have executed a 'continue' statement. */
    llvm::BasicBlock *continueTarget;

#ifdef ISPC_XE_ENABLED
    /** Final basic block of the function. It is used for Xe to
        disable returned lanes until return point is reached. */
    llvm::BasicBlock *returnPoint;
#endif

    /** @name Switch statement state

        These variables store various state that's active when we're
        generating code for a switch statement.  They should all be nullptr
        outside of a switch.
        @{
    */

    /** The value of the expression used to determine which case in the
        statements after the switch to execute. */
    llvm::Value *switchExpr;

    /** An AddressInfo with a pointer to memory that contains mask for lanes that should be
        active in the next block */
    AddressInfo *switchFallThroughMaskAddressInfo;

    /** Map from case label numbers to the basic block that will hold code
        for that case. */
    const std::vector<std::pair<int, llvm::BasicBlock *>> *caseBlocks;

    /** The basic block of code to run for the "default" label in the
        switch statement. */
    llvm::BasicBlock *defaultBlock;

    /** For each basic block for the code for cases (and the default label,
        if present), this map gives the basic block for the immediately
        following case/default label. */
    const std::map<llvm::BasicBlock *, llvm::BasicBlock *> *nextBlocks;

    /** Records whether the switch condition was uniform; this is a
        distinct notion from whether the switch represents uniform or
        varying control flow; we may have varying control flow from a
        uniform switch condition if there is a 'break' inside the switch
        that's under varying control flow. */
    bool switchConditionWasUniform;
    /** @} */

    /** AddressInfo with a pointer to memory that records which of the program instances
        have executed a 'return' statement (and are thus really truly done
        running any more instructions in this functions. */
    AddressInfo *returnedLanesAddressInfo;

    /** AddressInfo with a pointer to memory to store the return value for the function.
        Since difference program instances may execute 'return' statements
        at different times, we need to accumulate the return values as they
        come in until we return for real. */
    AddressInfo *returnValueAddressInfo;

    /** The CFInfo structure records information about a nesting level of
        control flow.  This vector lets us see what control flow is going
        around outside the current position in the function being
        emitted. */
    std::vector<CFInfo *> controlFlowInfo;

    /** DIFile object corresponding to the source file where the current
        function was defined (used for debugging info). */
    llvm::DIFile *diFile;

    /** DINamespace object corresponding to 'ispc' namespace in 'diFile'. */
    llvm::DINamespace *diSpace;

    /** DISubprogram corresponding to this function (used for debugging
        info). */
    llvm::DISubprogram *diSubprogram;

    /** These correspond to the current set of nested scopes in the
        function. */
    std::vector<llvm::DIScope *> debugScopes;

    /** True if a 'launch' statement has been encountered in the function. */
    bool launchedTasks;

    /** This is an AddressInfo with a pointer to a void * that is passed to the ISPCLaunch(),
        ISPCAlloc(), and ISPCSync() routines as a handle to the group ot
        tasks launched from the current function. */
    AddressInfo *launchGroupHandleAddressInfo;

    /** Nesting count of the number of times calling code has disabled (and
        not yet reenabled) gather/scatter performance warnings. */
    int disableGSWarningCount;

    std::map<std::string, llvm::BasicBlock *> labelMap;

    static bool initLabelBBlocks(ASTNode *node, void *data);

    llvm::Value *pointerVectorToVoidPointers(llvm::Value *value);
    static void addGSMetadata(llvm::Value *inst, SourcePos pos);
    bool ifsInCFAllUniform(int cfType) const;
    void jumpIfAllLoopLanesAreDone(llvm::BasicBlock *target);
    llvm::Value *emitGatherCallback(llvm::Value *lvalue, llvm::Value *retPtr);

    llvm::Value *applyVaryingGEP(llvm::Value *basePtr, llvm::Value *index, const Type *ptrType);

    void restoreMaskGivenReturns(llvm::Value *oldMask);
    void addSwitchMaskCheck(llvm::Value *mask);
    bool inSwitchStatement() const;
    llvm::Value *getMaskAtSwitchEntry();

    // Returns pointer to CFInfo object allocated on heap. This function
    // doesn't create or allocate this object. It removes CFInfo object from
    // controlFlowInfo vector and passes the ownership to the outer context.
    // The outer context should deconstruct this object.
    CFInfo *popCFState();

    void scatter(llvm::Value *value, llvm::Value *ptr, const Type *valueType, const Type *ptrType, llvm::Value *mask);
    void maskedStore(llvm::Value *value, llvm::Value *ptr, const Type *ptrType, llvm::Value *mask);
    void storeUniformToSOA(llvm::Value *value, llvm::Value *ptr, llvm::Value *mask, const Type *valueType,
                           const PointerType *ptrType);
    llvm::Value *loadUniformFromSOA(llvm::Value *ptr, llvm::Value *mask, const PointerType *ptrType,
                                    const llvm::Twine &name = "");

    llvm::Value *gather(llvm::Value *ptr, const PointerType *ptrType, llvm::Value *mask, const llvm::Twine &name = "");

    llvm::Value *addVaryingOffsetsIfNeeded(llvm::Value *ptr, const Type *ptrType);

    llvm::Value *lSwitchBoolSize_2(llvm::Value *value, llvm::Type *toType, bool toStorageType,
                                   const llvm::Twine &name = "");

    llvm::Value *lSwitchBoolSize_1(llvm::Value *value, llvm::Type *toType, bool toStorageType,
                                   const llvm::Twine &name = "");
};
} // namespace ispc
