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

/** @file stmt.cpp
    @brief File with definitions classes related to statements in the language
*/

#include "stmt.h"
#include "builtins-info.h"
#include "ctx.h"
#include "expr.h"
#include "func.h"
#include "llvmutil.h"
#include "module.h"
#include "sym.h"
#include "type.h"
#include "util.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <sstream>
#include <stdio.h>

#include <llvm/IR/CallingConv.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/raw_ostream.h>

using namespace ispc;

///////////////////////////////////////////////////////////////////////////
// Stmt

Stmt *Stmt::Optimize() { return this; }

void Stmt::SetLoopAttribute(std::pair<Globals::pragmaUnrollType, int> lAttr) {
    Error(pos, "Illegal pragma - expected a loop to follow '#pragma unroll/nounroll'.");
}

///////////////////////////////////////////////////////////////////////////
// ExprStmt

ExprStmt::ExprStmt(Expr *e, SourcePos p) : Stmt(p, ExprStmtID) { expr = e; }

void ExprStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock())
        return;

    ctx->SetDebugPos(pos);
    if (expr)
        expr->GetValue(ctx);
}

Stmt *ExprStmt::TypeCheck() { return this; }

void ExprStmt::Print(int indent) const {
    if (!expr)
        return;

    printf("%*c", indent, ' ');
    printf("Expr stmt: ");
    pos.Print();
    expr->Print();
    printf("\n");
}

int ExprStmt::EstimateCost() const { return 0; }

///////////////////////////////////////////////////////////////////////////
// DeclStmt

DeclStmt::DeclStmt(const std::vector<VariableDeclaration> &v, SourcePos p) : Stmt(p, DeclStmtID), vars(v) {}

static bool lHasUnsizedArrays(const Type *type) {
    const ArrayType *at = CastType<ArrayType>(type);
    if (at == NULL)
        return false;

    if (at->GetElementCount() == 0)
        return true;
    else
        return lHasUnsizedArrays(at->GetElementType());
}

void DeclStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock())
        return;

    for (unsigned int i = 0; i < vars.size(); ++i) {
        Symbol *sym = vars[i].sym;
        AssertPos(pos, sym != NULL);
        if (sym->type == NULL)
            continue;
        Expr *initExpr = vars[i].init;

        // Now that we're in the thick of emitting code, it's easy for us
        // to find out the level of nesting of varying control flow we're
        // in at this declaration.  So we can finally set that
        // Symbol::varyingCFDepth variable.
        // @todo It's disgusting to be doing this here.
        sym->varyingCFDepth = ctx->VaryingCFDepth();

        ctx->SetDebugPos(sym->pos);

        // If it's an array that was declared without a size but has an
        // initializer list, then use the number of elements in the
        // initializer list to finally set the array's size.
        sym->type = ArrayType::SizeUnsizedArrays(sym->type, initExpr);
        if (sym->type == NULL)
            continue;

        if (lHasUnsizedArrays(sym->type)) {
            Error(pos, "Illegal to declare an unsized array variable without "
                       "providing an initializer expression to set its size.");
            continue;
        }

        // References must have initializer expressions as well.
        if (IsReferenceType(sym->type) == true) {
            if (initExpr == NULL) {
                Error(sym->pos,
                      "Must provide initializer for reference-type "
                      "variable \"%s\".",
                      sym->name.c_str());
                continue;
            }
            if (IsReferenceType(initExpr->GetType()) == false) {
                const Type *initLVType = initExpr->GetLValueType();
                if (initLVType == NULL) {
                    Error(initExpr->pos,
                          "Initializer for reference-type variable "
                          "\"%s\" must have an lvalue type.",
                          sym->name.c_str());
                    continue;
                }
                if (initLVType->IsUniformType() == false) {
                    Error(initExpr->pos,
                          "Initializer for reference-type variable "
                          "\"%s\" must have a uniform lvalue type.",
                          sym->name.c_str());
                    continue;
                }
            }
        }

        llvm::Type *llvmType = sym->type->LLVMType(g->ctx);
        if (llvmType == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return;
        }

        if (sym->storageClass == SC_STATIC) {
            // For static variables, we need a compile-time constant value
            // for its initializer; if there's no initializer, we use a
            // zero value.
            llvm::Constant *cinit = NULL;
            if (initExpr != NULL) {
                if (PossiblyResolveFunctionOverloads(initExpr, sym->type) == false)
                    continue;
                // FIXME: we only need this for function pointers; it was
                // already done for atomic types and enums in
                // DeclStmt::TypeCheck()...
                if (llvm::dyn_cast<ExprList>(initExpr) == NULL) {
                    initExpr = TypeConvertExpr(initExpr, sym->type, "initializer");
                    // FIXME: and this is only needed to re-establish
                    // constant-ness so that GetConstant below works for
                    // constant artithmetic expressions...
                    initExpr = ::Optimize(initExpr);
                }

                std::pair<llvm::Constant *, bool> cinitPair = initExpr->GetConstant(sym->type);
                cinit = cinitPair.first;
                if (cinit == NULL)
                    Error(initExpr->pos,
                          "Initializer for static variable "
                          "\"%s\" must be a constant.",
                          sym->name.c_str());
            }
            if (cinit == NULL)
                cinit = llvm::Constant::getNullValue(llvmType);

            // Allocate space for the static variable in global scope, so
            // that it persists across function calls
            sym->storagePtr = new llvm::GlobalVariable(
                *m->module, llvmType, sym->type->IsConstType(), llvm::GlobalValue::InternalLinkage, cinit,
                llvm::Twine("static.") + llvm::Twine(sym->pos.first_line) + llvm::Twine(".") + sym->name.c_str());
            // Tell the FunctionEmitContext about the variable
            ctx->EmitVariableDebugInfo(sym);
        } else {
            // For non-static variables, allocate storage on the stack
            sym->storagePtr = ctx->AllocaInst(sym->type, sym->name.c_str());

            // Tell the FunctionEmitContext about the variable; must do
            // this before the initializer stuff.
            ctx->EmitVariableDebugInfo(sym);
            if (initExpr == 0 && sym->type->IsConstType())
                Error(sym->pos,
                      "Missing initializer for const variable "
                      "\"%s\".",
                      sym->name.c_str());

            // And then get it initialized...
            sym->parentFunction = ctx->GetFunction();
            InitSymbol(sym->storagePtr, sym->type, initExpr, ctx, sym->pos);
        }
    }
}

Stmt *DeclStmt::Optimize() {
    for (unsigned int i = 0; i < vars.size(); ++i) {
        Expr *init = vars[i].init;
        if (init != NULL && llvm::dyn_cast<ExprList>(init) == NULL) {
            // If the variable is const-qualified, after we've optimized
            // the initializer expression, see if we have a ConstExpr.  If
            // so, save it in Symbol::constValue where it can be used in
            // optimizing later expressions that have this symbol in them.
            // Note that there are cases where the expression may be
            // constant but where we don't have a ConstExpr; an example is
            // const arrays--the ConstExpr implementation just can't
            // represent an array of values.
            //
            // All this is fine in terms of the code that's generated in
            // the end (LLVM's constant folding stuff is good), but it
            // means that the ispc compiler's ability to reason about what
            // is definitely a compile-time constant for things like
            // computing array sizes from non-trivial expressions is
            // consequently limited.
            Symbol *sym = vars[i].sym;
            if (sym->type && sym->type->IsConstType() && Type::Equal(init->GetType(), sym->type))
                sym->constValue = llvm::dyn_cast<ConstExpr>(init);
        }
    }
    return this;
}

// Do type conversion if needed and check for not initializing array with
// another array (array assignment is not allowed).
// Do that recursively to handle brace initialization, which may contain
// another brace initialization.
static bool checkInit(const Type *type, Expr **init) {
    bool encounteredError = false;

    // get the right type for stuff like const float foo = 2; so that
    // the int->float type conversion is in there and we don't return
    // an int as the constValue later...
    if (CastType<AtomicType>(type) != NULL || CastType<EnumType>(type) != NULL) {
        // If it's an expr list with an atomic type, we'll later issue
        // an error.  Need to leave vars[i].init as is in that case so
        // it is in fact caught later, though.
        if (llvm::dyn_cast<ExprList>(*init) == NULL) {
            *init = TypeConvertExpr(*init, type, "initializer");
            if (*init == NULL)
                encounteredError = true;
        }
    } else if (CastType<ArrayType>(type) != NULL && llvm::dyn_cast<ExprList>(*init) == NULL) {
        encounteredError = true;
        Error((*init)->pos, "Array initializer must be an initializer list");
    } else if (CastType<StructType>(type) != NULL && llvm::dyn_cast<ExprList>(*init) != NULL) {
        const StructType *st = CastType<StructType>(type);
        ExprList *el = llvm::dyn_cast<ExprList>(*init);
        int elt_count = st->GetElementCount() < el->exprs.size() ? st->GetElementCount() : el->exprs.size();
        for (int i = 0; i < elt_count; i++) {
            encounteredError |= checkInit(st->GetElementType(i), &(el->exprs[i]));
        }
    }

    return encounteredError;
}

Stmt *DeclStmt::TypeCheck() {
    bool encounteredError = false;
    for (unsigned int i = 0; i < vars.size(); ++i) {
        if (vars[i].sym == NULL) {
            encounteredError = true;
            continue;
        }

        if (vars[i].init == NULL)
            continue;

        // Check an init.
        encounteredError |= checkInit(vars[i].sym->type, &(vars[i].init));
    }
    return encounteredError ? NULL : this;
}

void DeclStmt::Print(int indent) const {
    printf("%*cDecl Stmt:", indent, ' ');
    pos.Print();
    for (unsigned int i = 0; i < vars.size(); ++i) {
        printf("%*cVariable %s (%s)", indent + 4, ' ', vars[i].sym->name.c_str(),
               vars[i].sym->type->GetString().c_str());
        if (vars[i].init != NULL) {
            printf(" = ");
            vars[i].init->Print();
        }
        printf("\n");
    }
    printf("\n");
}

int DeclStmt::EstimateCost() const { return 0; }

///////////////////////////////////////////////////////////////////////////
// IfStmt

IfStmt::IfStmt(Expr *t, Stmt *ts, Stmt *fs, bool checkCoherence, SourcePos p)
    : Stmt(p, IfStmtID), test(t), trueStmts(ts), falseStmts(fs),
      doAllCheck(checkCoherence && !g->opt.disableCoherentControlFlow) {}

static void lEmitIfStatements(FunctionEmitContext *ctx, Stmt *stmts, const char *trueOrFalse) {
    if (!stmts)
        return;

    if (llvm::dyn_cast<StmtList>(stmts) == NULL)
        ctx->StartScope();
    ctx->AddInstrumentationPoint(trueOrFalse);
    stmts->EmitCode(ctx);
    if (llvm::dyn_cast<const StmtList>(stmts) == NULL)
        ctx->EndScope();
}

/** Returns true if the "true" block for the if statement consists of a
    single 'break' statement, and the "false" block is empty. */
/*
static bool
lCanApplyBreakOptimization(Stmt *trueStmts, Stmt *falseStmts) {
    if (falseStmts != NULL) {
        if (StmtList *sl = llvm::dyn_cast<StmtList>(falseStmts)) {
            return (sl->stmts.size() == 0);
        }
        else
            return false;
    }

    if (llvm::dyn_cast<BreakStmt>(trueStmts))
        return true;
    else if (StmtList *sl = llvm::dyn_cast<StmtList>(trueStmts))
        return (sl->stmts.size() == 1 &&
                llvm::dyn_cast<BreakStmt>(sl->stmts[0]) != NULL);
    else
        return false;
}
*/

void IfStmt::EmitCode(FunctionEmitContext *ctx) const {
    // First check all of the things that might happen due to errors
    // earlier in compilation and bail out if needed so that we don't
    // dereference NULL pointers in the below...
    if (!ctx->GetCurrentBasicBlock())
        return;
    if (!test)
        return;
    const Type *testType = test->GetType();
    if (!testType)
        return;

    ctx->SetDebugPos(pos);
    bool isUniform = testType->IsUniformType();

    llvm::Value *testValue = test->GetValue(ctx);
    if (testValue == NULL)
        return;

    bool emulateUniform = false;
    if (ctx->emitXeHardwareMask() && !isUniform) {
        /* With Xe target we generate uniform control flow but
           emit varying using CM simdcf.any intrinsic. We mark the scope as
           emulateUniform = true to let nested scopes know that they should
           generate vector conditions before branching.
           This is needed because CM does not support scalar control flow inside
           simd control flow.
         */
        isUniform = true;
        emulateUniform = true;
    }

    if (isUniform) {
        ctx->StartUniformIf(emulateUniform);
        if (doAllCheck && !emulateUniform)
            Warning(test->pos, "Uniform condition supplied to \"cif\" statement.");

        // 'If' statements with uniform conditions are relatively
        // straightforward.  We evaluate the condition and then jump to
        // either the 'then' or 'else' clause depending on its value.
        llvm::BasicBlock *bthen = ctx->CreateBasicBlock("if_then", ctx->GetCurrentBasicBlock());
        llvm::BasicBlock *belse = ctx->CreateBasicBlock("if_else", bthen);
        llvm::BasicBlock *bexit = ctx->CreateBasicBlock("if_exit", belse);

        // Jump to the appropriate basic block based on the value of
        // the 'if' test
        ctx->BranchInst(bthen, belse, testValue);

        // Emit code for the 'true' case
        ctx->SetCurrentBasicBlock(bthen);
        lEmitIfStatements(ctx, trueStmts, "true");
        if (ctx->GetCurrentBasicBlock())
            ctx->BranchInst(bexit);

        // Emit code for the 'false' case
        ctx->SetCurrentBasicBlock(belse);
        lEmitIfStatements(ctx, falseStmts, "false");
        if (ctx->GetCurrentBasicBlock())
            ctx->BranchInst(bexit);

        // Set the active basic block to the newly-created exit block
        // so that subsequent emitted code starts there.
        ctx->SetCurrentBasicBlock(bexit);
        ctx->EndIf();
    }
    /*
    // Disabled for performance reasons.  Change to an optional compile-time opt switch.
    else if (lCanApplyBreakOptimization(trueStmts, falseStmts)) {
        // If we have a simple break statement inside the 'if' and are
        // under varying control flow, just update the execution mask
        // directly and don't emit code for the statements.  This leads to
        // better code for this case--this is surprising and should be
        // root-caused further, but for now this gives us performance
        // benefit in this case.
        ctx->SetInternalMaskAndNot(ctx->GetInternalMask(), testValue);
    }
    */
    else
        emitVaryingIf(ctx, testValue);
}

Stmt *IfStmt::TypeCheck() {
    if (test != NULL) {
        const Type *testType = test->GetType();
        if (testType != NULL) {
            bool isUniform = (testType->IsUniformType() && !g->opt.disableUniformControlFlow);
            test = TypeConvertExpr(test, isUniform ? AtomicType::UniformBool : AtomicType::VaryingBool,
                                   "\"if\" statement test");
            if (test == NULL)
                return NULL;
        }
    }

    return this;
}

int IfStmt::EstimateCost() const {
    const Type *type;
    if (test == NULL || (type = test->GetType()) == NULL)
        return 0;

    return type->IsUniformType() ? COST_UNIFORM_IF : COST_VARYING_IF;
}

void IfStmt::Print(int indent) const {
    printf("%*cIf Stmt %s", indent, ' ', doAllCheck ? "DO ALL CHECK" : "");
    pos.Print();
    printf("\n%*cTest: ", indent + 4, ' ');
    test->Print();
    printf("\n");
    if (trueStmts) {
        printf("%*cTrue:\n", indent + 4, ' ');
        trueStmts->Print(indent + 8);
    }
    if (falseStmts) {
        printf("%*cFalse:\n", indent + 4, ' ');
        falseStmts->Print(indent + 8);
    }
}

/** Emit code to run both the true and false statements for the if test,
    with the mask set appropriately before running each one.
*/
void IfStmt::emitMaskedTrueAndFalse(FunctionEmitContext *ctx, llvm::Value *oldMask, llvm::Value *test) const {
    if (trueStmts) {
        ctx->SetInternalMaskAnd(oldMask, test);
        lEmitIfStatements(ctx, trueStmts, "if: expr mixed, true statements");
        // under varying control flow,, returns can't stop instruction
        // emission, so this better be non-NULL...
        AssertPos(ctx->GetDebugPos(), ctx->GetCurrentBasicBlock());
    }
    if (falseStmts) {
        ctx->SetInternalMaskAndNot(oldMask, test);
        lEmitIfStatements(ctx, falseStmts, "if: expr mixed, false statements");
        AssertPos(ctx->GetDebugPos(), ctx->GetCurrentBasicBlock());
    }
}

/** Emit code for an if test that checks the mask and the test values and
    tries to be smart about jumping over code that doesn't need to be run.
 */
void IfStmt::emitVaryingIf(FunctionEmitContext *ctx, llvm::Value *ltest) const {
    llvm::Value *oldMask = ctx->GetInternalMask();
    if (doAllCheck) {
        // We can't tell if the mask going into the if is all on at the
        // compile time.  Emit code to check for this and then either run
        // the code for the 'all on' or the 'mixed' case depending on the
        // mask's value at runtime.
        llvm::BasicBlock *bAllOn = ctx->CreateBasicBlock("cif_mask_all");
        llvm::BasicBlock *bMixedOn = ctx->CreateBasicBlock("cif_mask_mixed");
        llvm::BasicBlock *bDone = ctx->CreateBasicBlock("cif_done");

        // Jump to either bAllOn or bMixedOn, depending on the mask's value
        llvm::Value *maskAllQ = ctx->All(ctx->GetFullMask());
        ctx->BranchInst(bAllOn, bMixedOn, maskAllQ);

        // Emit code for the 'mask all on' case
        ctx->SetCurrentBasicBlock(bAllOn);
        emitMaskAllOn(ctx, ltest, bDone);

        // And emit code for the mixed mask case
        ctx->SetCurrentBasicBlock(bMixedOn);
        emitMaskMixed(ctx, oldMask, ltest, bDone);

        // When done, set the current basic block to the block that the two
        // paths above jump to when they're done.
        ctx->SetCurrentBasicBlock(bDone);
    } else if (trueStmts != NULL || falseStmts != NULL) {
        // If there is nothing that is potentially unsafe to run with all
        // lanes off in the true and false statements and if the total
        // complexity of those two is relatively simple, then we'll go
        // ahead and emit straightline code that runs both sides, updating
        // the mask accordingly.  This is useful for efficiently compiling
        // things like:
        //
        // if (foo) x = 0;
        // else     ++x;
        //
        // Where the overhead of checking if any of the program instances wants
        // to run one side or the other is more than the actual computation.
        // SafeToRunWithMaskAllOff() checks to make sure that we don't do this
        // for potentially dangerous code like:
        //
        // if (index < count) array[index] = 0;
        //
        // where our use of blend for conditional assignments doesn't check
        // for the 'all lanes' off case.
        int trueFalseCost = (::EstimateCost(trueStmts) + ::EstimateCost(falseStmts));
        bool costIsAcceptable = (trueFalseCost < PREDICATE_SAFE_IF_STATEMENT_COST);

        bool safeToRunWithAllLanesOff = (SafeToRunWithMaskAllOff(trueStmts) && SafeToRunWithMaskAllOff(falseStmts));

        Debug(pos, "If statement: true cost %d (safe %d), false cost %d (safe %d).", ::EstimateCost(trueStmts),
              (int)SafeToRunWithMaskAllOff(trueStmts), ::EstimateCost(falseStmts),
              (int)SafeToRunWithMaskAllOff(falseStmts));

        if (safeToRunWithAllLanesOff && (costIsAcceptable || g->opt.disableCoherentControlFlow)) {
            ctx->StartVaryingIf(oldMask);
            emitMaskedTrueAndFalse(ctx, oldMask, ltest);
            AssertPos(pos, ctx->GetCurrentBasicBlock());
            ctx->EndIf();
        } else {
            llvm::BasicBlock *bDone = ctx->CreateBasicBlock("if_done");
            emitMaskMixed(ctx, oldMask, ltest, bDone);
            ctx->SetCurrentBasicBlock(bDone);
        }
    }
}

/** Emits code for 'if' tests under the case where we know that the program
    mask is all on going into the 'if'.
 */
void IfStmt::emitMaskAllOn(FunctionEmitContext *ctx, llvm::Value *ltest, llvm::BasicBlock *bDone) const {
    // We start by explicitly storing "all on" into the mask mask.  Note
    // that this doesn't change its actual value, but doing so lets the
    // compiler see what's going on so that subsequent optimizations for
    // code emitted here can operate with the knowledge that the mask is
    // definitely all on (until it modifies the mask itself).
    AssertPos(pos, !g->opt.disableCoherentControlFlow);
    if (!g->opt.disableMaskAllOnOptimizations)
        ctx->SetInternalMask(LLVMMaskAllOn);
    llvm::Value *oldFunctionMask = ctx->GetFunctionMask();
    if (!g->opt.disableMaskAllOnOptimizations)
        ctx->SetFunctionMask(LLVMMaskAllOn);

    // First, check the value of the test.  If it's all on, then we jump to
    // a basic block that will only have code for the true case.
    llvm::BasicBlock *bTestAll = ctx->CreateBasicBlock("cif_test_all");
    llvm::BasicBlock *bTestNoneCheck = ctx->CreateBasicBlock("cif_test_none_check");
    llvm::Value *testAllQ = ctx->All(ltest);
    ctx->BranchInst(bTestAll, bTestNoneCheck, testAllQ);

    // Emit code for the 'test is all true' case
    ctx->SetCurrentBasicBlock(bTestAll);
    ctx->StartVaryingIf(LLVMMaskAllOn);
    lEmitIfStatements(ctx, trueStmts, "if: all on mask, expr all true");
    ctx->EndIf();
    if (ctx->GetCurrentBasicBlock() != NULL)
        // bblock may legitimately be NULL since if there's a return stmt
        // or break or continue we can actually jump and end emission since
        // we know all of the lanes are following this path...
        ctx->BranchInst(bDone);

    // The test isn't all true.  Now emit code to determine if it's all
    // false, or has mixed values.
    ctx->SetCurrentBasicBlock(bTestNoneCheck);
    llvm::BasicBlock *bTestNone = ctx->CreateBasicBlock("cif_test_none");
    llvm::BasicBlock *bTestMixed = ctx->CreateBasicBlock("cif_test_mixed");
    llvm::Value *testMixedQ = ctx->Any(ltest);
    ctx->BranchInst(bTestMixed, bTestNone, testMixedQ);

    // Emit code for the 'test is all false' case
    ctx->SetCurrentBasicBlock(bTestNone);
    ctx->StartVaryingIf(LLVMMaskAllOn);
    lEmitIfStatements(ctx, falseStmts, "if: all on mask, expr all false");
    ctx->EndIf();
    if (ctx->GetCurrentBasicBlock())
        // bblock may be NULL since if there's a return stmt or break or
        // continue we can actually jump or whatever and end emission...
        ctx->BranchInst(bDone);

    // Finally emit code for the 'mixed true/false' case.  We unavoidably
    // need to run both the true and the false statements.
    ctx->SetCurrentBasicBlock(bTestMixed);
    ctx->StartVaryingIf(LLVMMaskAllOn);
    emitMaskedTrueAndFalse(ctx, LLVMMaskAllOn, ltest);
    // In this case, return/break/continue isn't allowed to jump and end
    // emission.
    AssertPos(pos, ctx->GetCurrentBasicBlock());
    ctx->EndIf();
    ctx->BranchInst(bDone);

    ctx->SetCurrentBasicBlock(bDone);
    ctx->SetFunctionMask(oldFunctionMask);
}

/** Emit code for an 'if' test where the lane mask is known to be mixed
    on/off going into it.
 */
void IfStmt::emitMaskMixed(FunctionEmitContext *ctx, llvm::Value *oldMask, llvm::Value *ltest,
                           llvm::BasicBlock *bDone) const {
    ctx->StartVaryingIf(oldMask);
    llvm::BasicBlock *bNext = ctx->CreateBasicBlock("safe_if_after_true");

    llvm::BasicBlock *bRunTrue = ctx->CreateBasicBlock("safe_if_run_true");
    ctx->SetInternalMaskAnd(oldMask, ltest);

    // Do any of the program instances want to run the 'true'
    // block?  If not, jump ahead to bNext.

    llvm::Value *maskAnyTrueQ = ctx->Any(ctx->GetFullMask());

    ctx->BranchInst(bRunTrue, bNext, maskAnyTrueQ);

    // Emit statements for true
    ctx->SetCurrentBasicBlock(bRunTrue);
    if (trueStmts != NULL)
        lEmitIfStatements(ctx, trueStmts, "if: expr mixed, true statements");
    AssertPos(pos, ctx->GetCurrentBasicBlock());
    ctx->BranchInst(bNext);
    ctx->SetCurrentBasicBlock(bNext);

    // False...
    llvm::BasicBlock *bRunFalse = ctx->CreateBasicBlock("safe_if_run_false");
    ctx->SetInternalMaskAndNot(oldMask, ltest);

    // Similarly, check to see if any of the instances want to
    // run the 'false' block...

    llvm::Value *maskAnyFalseQ = ctx->Any(ctx->GetFullMask());
    ctx->BranchInst(bRunFalse, bDone, maskAnyFalseQ);

    // Emit code for false
    ctx->SetCurrentBasicBlock(bRunFalse);
    if (falseStmts)
        lEmitIfStatements(ctx, falseStmts, "if: expr mixed, false statements");
    AssertPos(pos, ctx->GetCurrentBasicBlock());

    ctx->BranchInst(bDone);
    ctx->SetCurrentBasicBlock(bDone);
    ctx->EndIf();
}

///////////////////////////////////////////////////////////////////////////
// DoStmt

struct VaryingBCCheckInfo {
    VaryingBCCheckInfo() {
        varyingControlFlowDepth = 0;
        foundVaryingBreakOrContinue = false;
    }

    int varyingControlFlowDepth;
    bool foundVaryingBreakOrContinue;
};

/** Returns true if the given node is an 'if' statement where the test
    condition has varying type. */
static bool lIsVaryingFor(ASTNode *node) {
    IfStmt *ifStmt;
    if ((ifStmt = llvm::dyn_cast<IfStmt>(node)) != NULL && ifStmt->test != NULL) {
        const Type *type = ifStmt->test->GetType();
        return (type != NULL && type->IsVaryingType());
    } else
        return false;
}

/** Preorder callback function for checking for varying breaks or
    continues. */
static bool lVaryingBCPreFunc(ASTNode *node, void *d) {
    VaryingBCCheckInfo *info = (VaryingBCCheckInfo *)d;

    // We found a break or continue statement; if we're under varying
    // control flow, then bingo.
    if ((llvm::dyn_cast<BreakStmt>(node) != NULL || llvm::dyn_cast<ContinueStmt>(node) != NULL) &&
        info->varyingControlFlowDepth > 0) {
        info->foundVaryingBreakOrContinue = true;
        return false;
    }

    // Update the count of the nesting depth of varying control flow if
    // this is an if statement with a varying condition.
    if (lIsVaryingFor(node))
        ++info->varyingControlFlowDepth;

    if (llvm::dyn_cast<ForStmt>(node) != NULL || llvm::dyn_cast<DoStmt>(node) != NULL ||
        llvm::dyn_cast<ForeachStmt>(node) != NULL)
        // Don't recurse into these guys, since we don't care about varying
        // breaks or continues within them...
        return false;
    else
        return true;
}

/** Postorder callback function for checking for varying breaks or
    continues; decrement the varying control flow depth after the node's
    children have been processed, if this is a varying if statement. */
static ASTNode *lVaryingBCPostFunc(ASTNode *node, void *d) {
    VaryingBCCheckInfo *info = (VaryingBCCheckInfo *)d;
    if (lIsVaryingFor(node))
        --info->varyingControlFlowDepth;
    return node;
}

/** Given a statment, walk through it to see if there is a 'break' or
    'continue' statement inside if its children, under varying control
    flow.  We need to detect this case for loops since what might otherwise
    look like a 'uniform' loop needs to have code emitted to do all of the
    lane management stuff if this is the case.
 */
static bool lHasVaryingBreakOrContinue(Stmt *stmt) {
    VaryingBCCheckInfo info;
    WalkAST(stmt, lVaryingBCPreFunc, lVaryingBCPostFunc, &info);
    return info.foundVaryingBreakOrContinue;
}

DoStmt::DoStmt(Expr *t, Stmt *s, bool cc, SourcePos p)
    : Stmt(p, DoStmtID), testExpr(t), bodyStmts(s), doCoherentCheck(cc && !g->opt.disableCoherentControlFlow) {}

void DoStmt::EmitCode(FunctionEmitContext *ctx) const {
    // Check for things that could be NULL due to earlier errors during
    // compilation.
    if (!ctx->GetCurrentBasicBlock())
        return;
    if (!testExpr || !testExpr->GetType())
        return;

    bool uniformTest = testExpr->GetType()->IsUniformType();

    if (uniformTest && doCoherentCheck)
        Warning(testExpr->pos, "Uniform condition supplied to \"cdo\" "
                               "statement.");

    llvm::BasicBlock *bloop = ctx->CreateBasicBlock("do_loop", ctx->GetCurrentBasicBlock());
    llvm::BasicBlock *btest = ctx->CreateBasicBlock("do_test", bloop);
    llvm::BasicBlock *bexit = ctx->CreateBasicBlock("do_exit", btest);
    bool emulateUniform = false;
    llvm::Instruction *branchInst = NULL;
    if (ctx->emitXeHardwareMask() && !uniformTest) {
        /* With Xe target we generate uniform control flow but
           emit varying using CM simdcf.any intrinsic. We mark the scope as
           emulateUniform = true to let nested scopes know that they should
           generate vector conditions before branching.
           This is needed because CM does not support scalar control flow inside
           simd control flow.
         */
        uniformTest = true;
        emulateUniform = true;
    }
    ctx->StartLoop(bexit, btest, uniformTest, emulateUniform);

    // Start by jumping into the loop body
    ctx->BranchInst(bloop);

    // And now emit code for the loop body
    ctx->SetCurrentBasicBlock(bloop);
    ctx->SetBlockEntryMask(ctx->GetFullMask());
    ctx->SetDebugPos(pos);
    // FIXME: in the StmtList::EmitCode() method takes starts/stops a new
    // scope around the statements in the list.  So if the body is just a
    // single statement (and thus not a statement list), we need a new
    // scope, but we don't want two scopes in the StmtList case.
    if (!bodyStmts || !llvm::dyn_cast<StmtList>(bodyStmts))
        ctx->StartScope();

    ctx->AddInstrumentationPoint("do loop body");
    if (doCoherentCheck && !uniformTest) {
        // Check to see if the mask is all on
        llvm::BasicBlock *bAllOn = ctx->CreateBasicBlock("do_all_on");
        llvm::BasicBlock *bMixed = ctx->CreateBasicBlock("do_mixed");
        ctx->BranchIfMaskAll(bAllOn, bMixed);

        // If so, emit code for the 'mask all on' case.  In particular,
        // explicitly set the mask to 'all on' (see rationale in
        // IfStmt::emitCoherentTests()), and then emit the code for the
        // loop body.
        ctx->SetCurrentBasicBlock(bAllOn);
        if (!g->opt.disableMaskAllOnOptimizations)
            ctx->SetInternalMask(LLVMMaskAllOn);
        llvm::Value *oldFunctionMask = ctx->GetFunctionMask();
        if (!g->opt.disableMaskAllOnOptimizations)
            ctx->SetFunctionMask(LLVMMaskAllOn);
        if (bodyStmts)
            bodyStmts->EmitCode(ctx);
        AssertPos(pos, ctx->GetCurrentBasicBlock());
        ctx->SetFunctionMask(oldFunctionMask);
        ctx->BranchInst(btest);

        // The mask is mixed.  Just emit the code for the loop body.
        ctx->SetCurrentBasicBlock(bMixed);
        if (bodyStmts)
            bodyStmts->EmitCode(ctx);
        AssertPos(pos, ctx->GetCurrentBasicBlock());
        ctx->BranchInst(btest);
    } else {
        // Otherwise just emit the code for the loop body.  The current
        // mask is good.
        if (bodyStmts)
            bodyStmts->EmitCode(ctx);
        if (ctx->GetCurrentBasicBlock()) {
            ctx->BranchInst(btest);
        }
    }
    // End the scope we started above, if needed.
    if (!bodyStmts || !llvm::dyn_cast<StmtList>(bodyStmts))
        ctx->EndScope();

    // Now emit code for the loop test.
    ctx->SetCurrentBasicBlock(btest);
    // First, emit code to restore the mask value for any lanes that
    // executed a 'continue' during the current loop before we go and emit
    // the code for the test.  This is only necessary for varying loops;
    // 'uniform' loops just jump when they hit a continue statement and
    // don't mess with the mask.
    if (!uniformTest) {
        ctx->RestoreContinuedLanes();
        ctx->ClearBreakLanes();
    }
    llvm::Value *testValue = testExpr->GetValue(ctx);
    if (!testValue)
        return;

    if (uniformTest) {
        // For the uniform case, just jump to the top of the loop or the
        // exit basic block depending on the value of the test.
        branchInst = ctx->BranchInst(bloop, bexit, testValue);
        ctx->setLoopUnrollMetadata(branchInst, loopAttribute, pos);
    } else {
        // For the varying case, update the mask based on the value of the
        // test.  If any program instances still want to be running, jump
        // to the top of the loop.  Otherwise, jump out.
        llvm::Value *mask = ctx->GetInternalMask();
        ctx->SetInternalMaskAnd(mask, testValue);
        ctx->BranchIfMaskAny(bloop, bexit);
    }

    // ...and we're done.  Set things up for subsequent code to be emitted
    // in the right basic block.
    ctx->SetCurrentBasicBlock(bexit);
    ctx->EndLoop();
}

Stmt *DoStmt::TypeCheck() {
    const Type *testType;
    if (testExpr != NULL && (testType = testExpr->GetType()) != NULL) {
        // Should the test condition for the loop be uniform or varying?
        // It can be uniform only if three conditions are met:
        //
        // - First and foremost, the type of the test condition must be
        //   uniform.
        //
        // - Second, the user must not have set the dis-optimization option
        //   that disables uniform flow control.
        //
        // - Thirdly, and most subtlely, there must not be any break or
        //   continue statements inside the loop that are within the scope
        //   of a 'varying' if statement.  If there are, then we type cast
        //   the test to be 'varying', so that the code generated for the
        //   loop includes masking stuff, so that we can track which lanes
        //   actually want to be running, accounting for breaks/continues.
        //
        bool uniformTest =
            (testType->IsUniformType() && !g->opt.disableUniformControlFlow && !lHasVaryingBreakOrContinue(bodyStmts));
        testExpr = TypeConvertExpr(testExpr, uniformTest ? AtomicType::UniformBool : AtomicType::VaryingBool,
                                   "\"do\" statement");
    }

    return this;
}

void DoStmt::SetLoopAttribute(std::pair<Globals::pragmaUnrollType, int> lAttr) {
    if (loopAttribute.first != Globals::pragmaUnrollType::none)
        Error(pos, "Multiple '#pragma unroll/nounroll' directives used.");
    bool uniformTest = testExpr ? testExpr->GetType()->IsUniformType()
                                : (!g->opt.disableUniformControlFlow && !lHasVaryingBreakOrContinue(bodyStmts));
    if (uniformTest) {
        loopAttribute = lAttr;
    } else {
        Warning(pos, "'#pragma unroll/nounroll' ignored - not supported for varying do loop.");
    }
}

int DoStmt::EstimateCost() const {
    bool uniformTest = testExpr ? testExpr->GetType()->IsUniformType()
                                : (!g->opt.disableUniformControlFlow && !lHasVaryingBreakOrContinue(bodyStmts));

    return uniformTest ? COST_UNIFORM_LOOP : COST_VARYING_LOOP;
}

void DoStmt::Print(int indent) const {
    printf("%*cDo Stmt", indent, ' ');
    pos.Print();
    printf(":\n");
    printf("%*cTest: ", indent + 4, ' ');
    if (testExpr)
        testExpr->Print();
    printf("\n");
    if (bodyStmts) {
        printf("%*cStmts:\n", indent + 4, ' ');
        bodyStmts->Print(indent + 8);
    }
}

///////////////////////////////////////////////////////////////////////////
// ForStmt

ForStmt::ForStmt(Stmt *i, Expr *t, Stmt *s, Stmt *st, bool cc, SourcePos p)
    : Stmt(p, ForStmtID), init(i), test(t), step(s), stmts(st),
      doCoherentCheck(cc && !g->opt.disableCoherentControlFlow) {}

void ForStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock())
        return;

    llvm::BasicBlock *btest = ctx->CreateBasicBlock("for_test", ctx->GetCurrentBasicBlock());
    llvm::BasicBlock *bloop = ctx->CreateBasicBlock("for_loop", btest);
    llvm::BasicBlock *bstep = ctx->CreateBasicBlock("for_step", bloop);
    llvm::BasicBlock *bexit = ctx->CreateBasicBlock("for_exit", bstep);

    bool uniformTest = test ? test->GetType()->IsUniformType()
                            : (!g->opt.disableUniformControlFlow && !lHasVaryingBreakOrContinue(stmts));
    bool emulateUniform = false;
    if (ctx->emitXeHardwareMask() && !uniformTest) {
        /* With Xe target we generate uniform control flow but
           emit varying using CM simdcf.any intrinsic. We mark the scope as
           emulateUniform = true to let nested scopes know that they should
           generate vector conditions before branching.
           This is needed because CM does not support scalar control flow inside
           simd control flow.
         */
        uniformTest = true;
        emulateUniform = true;
    }
    ctx->StartLoop(bexit, bstep, uniformTest, emulateUniform);
    ctx->SetDebugPos(pos);

    // If we have an initiailizer statement, start by emitting the code for
    // it and then jump into the loop test code.  (Also start a new scope
    // since the initiailizer may be a declaration statement).
    if (init) {
        AssertPos(pos, llvm::dyn_cast<StmtList>(init) == NULL);
        ctx->StartScope();
        init->EmitCode(ctx);
    }
    ctx->BranchInst(btest);

    // Emit code to get the value of the loop test.  If no test expression
    // was provided, just go with a true value.

    ctx->SetCurrentBasicBlock(btest);
    llvm::Value *ltest = NULL;
    if (test) {
        ltest = test->GetValue(ctx);
        if (!ltest) {
            // We need to end scope only if we had initializer statement.
            if (init) {
                ctx->EndScope();
            }
            ctx->EndLoop();
            return;
        }
    } else
        ltest = uniformTest ? LLVMTrue : LLVMBoolVector(true);
    // Now use the test's value.  For a uniform loop, we can either jump to
    // the loop body or the loop exit, based on whether it's true or false.
    // For a non-uniform loop, we update the mask and jump into the loop if
    // any of the mask values are true.
    if (uniformTest) {
        if (doCoherentCheck && !emulateUniform)
            if (test)
                Warning(test->pos, "Uniform condition supplied to cfor/cwhile "
                                   "statement.");
        if (!ctx->emitXeHardwareMask())
            AssertPos(pos, ltest->getType() == LLVMTypes::BoolType);
        ctx->BranchInst(bloop, bexit, ltest);
    } else {
        llvm::Value *mask = ctx->GetInternalMask();
        ctx->SetInternalMaskAnd(mask, ltest);
        ctx->BranchIfMaskAny(bloop, bexit);
    }

    // On to emitting the code for the loop body.
    ctx->SetCurrentBasicBlock(bloop);
    ctx->SetBlockEntryMask(ctx->GetFullMask());
    ctx->AddInstrumentationPoint("for loop body");
    if (!llvm::dyn_cast_or_null<StmtList>(stmts))
        ctx->StartScope();

    if (doCoherentCheck && !uniformTest) {
        // For 'varying' loops with the coherence check, we start by
        // checking to see if the mask is all on, after it has been updated
        // based on the value of the test.
        llvm::BasicBlock *bAllOn = ctx->CreateBasicBlock("for_all_on");
        llvm::BasicBlock *bMixed = ctx->CreateBasicBlock("for_mixed");
        ctx->BranchIfMaskAll(bAllOn, bMixed);

        // Emit code for the mask being all on.  Explicitly set the mask to
        // be on so that the optimizer can see that it's on (i.e. now that
        // the runtime test has passed, make this fact clear for code
        // generation at compile time here.)
        ctx->SetCurrentBasicBlock(bAllOn);
        if (!g->opt.disableMaskAllOnOptimizations)
            ctx->SetInternalMask(LLVMMaskAllOn);
        llvm::Value *oldFunctionMask = ctx->GetFunctionMask();
        if (!g->opt.disableMaskAllOnOptimizations)
            ctx->SetFunctionMask(LLVMMaskAllOn);
        if (stmts)
            stmts->EmitCode(ctx);
        AssertPos(pos, ctx->GetCurrentBasicBlock());
        ctx->SetFunctionMask(oldFunctionMask);
        ctx->BranchInst(bstep);

        // Emit code for the mask being mixed.  We should never run the
        // loop with the mask all off, based on the BranchIfMaskAny call
        // above.
        ctx->SetCurrentBasicBlock(bMixed);
        if (stmts)
            stmts->EmitCode(ctx);
        ctx->BranchInst(bstep);
    } else {
        // For both uniform loops and varying loops without the coherence
        // check, we know that at least one program instance wants to be
        // running the loop, so just emit code for the loop body and jump
        // to the loop step code.
        if (stmts)
            stmts->EmitCode(ctx);
        if (ctx->GetCurrentBasicBlock())
            ctx->BranchInst(bstep);
    }
    if (!llvm::dyn_cast_or_null<StmtList>(stmts))
        ctx->EndScope();

    // Emit code for the loop step.  First, restore the lane mask of any
    // program instances that executed a 'continue' during the previous
    // iteration.  Then emit code for the loop step and then jump to the
    // test code.
    ctx->SetCurrentBasicBlock(bstep);
    ctx->RestoreContinuedLanes();
    ctx->ClearBreakLanes();

    if (step)
        step->EmitCode(ctx);

    llvm::Instruction *branchInst = ctx->BranchInst(btest);
    if (uniformTest) {
        ctx->setLoopUnrollMetadata(branchInst, loopAttribute, pos);
    }

    // Set the current emission basic block to the loop exit basic block
    ctx->SetCurrentBasicBlock(bexit);
    if (init)
        ctx->EndScope();
    ctx->EndLoop();
}

Stmt *ForStmt::TypeCheck() {
    const Type *testType;
    if (test && (testType = test->GetType()) != NULL) {
        // See comments in DoStmt::TypeCheck() regarding
        // 'uniformTest' and the type conversion here.
        bool uniformTest =
            (testType->IsUniformType() && !g->opt.disableUniformControlFlow && !lHasVaryingBreakOrContinue(stmts));
        test = TypeConvertExpr(test, uniformTest ? AtomicType::UniformBool : AtomicType::VaryingBool,
                               "\"for\"/\"while\" statement");
        if (test == NULL)
            return NULL;
    }

    return this;
}

void ForStmt::SetLoopAttribute(std::pair<Globals::pragmaUnrollType, int> lAttr) {
    if (loopAttribute.first != Globals::pragmaUnrollType::none)
        Error(pos, "Multiple '#pragma unroll/nounroll' directives used.");

    bool uniformTest = test ? test->GetType()->IsUniformType()
                            : (!g->opt.disableUniformControlFlow && !lHasVaryingBreakOrContinue(stmts));

    if (uniformTest) {
        loopAttribute = lAttr;
    } else {
        Warning(pos, "'#pragma unroll/nounroll' ignored - not supported for varying for loop.");
    }
}

int ForStmt::EstimateCost() const {
    bool uniformTest = test ? test->GetType()->IsUniformType()
                            : (!g->opt.disableUniformControlFlow && !lHasVaryingBreakOrContinue(stmts));

    return uniformTest ? COST_UNIFORM_LOOP : COST_VARYING_LOOP;
}

void ForStmt::Print(int indent) const {
    printf("%*cFor Stmt", indent, ' ');
    pos.Print();
    printf("\n");
    if (init) {
        printf("%*cInit:\n", indent + 4, ' ');
        init->Print(indent + 8);
    }
    if (test) {
        printf("%*cTest: ", indent + 4, ' ');
        test->Print();
        printf("\n");
    }
    if (step) {
        printf("%*cStep:\n", indent + 4, ' ');
        step->Print(indent + 8);
    }
    if (stmts) {
        printf("%*cStmts:\n", indent + 4, ' ');
        stmts->Print(indent + 8);
    }
}

///////////////////////////////////////////////////////////////////////////
// BreakStmt

BreakStmt::BreakStmt(SourcePos p) : Stmt(p, BreakStmtID) {}

void BreakStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock())
        return;

    ctx->SetDebugPos(pos);
    ctx->Break(true);
}

Stmt *BreakStmt::TypeCheck() { return this; }

int BreakStmt::EstimateCost() const { return COST_BREAK_CONTINUE; }

void BreakStmt::Print(int indent) const {
    printf("%*cBreak Stmt", indent, ' ');
    pos.Print();
    printf("\n");
}

///////////////////////////////////////////////////////////////////////////
// ContinueStmt

ContinueStmt::ContinueStmt(SourcePos p) : Stmt(p, ContinueStmtID) {}

void ContinueStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock())
        return;

    ctx->SetDebugPos(pos);
    ctx->Continue(true);
}

Stmt *ContinueStmt::TypeCheck() { return this; }

int ContinueStmt::EstimateCost() const { return COST_BREAK_CONTINUE; }

void ContinueStmt::Print(int indent) const {
    printf("%*cContinue Stmt", indent, ' ');
    pos.Print();
    printf("\n");
}

///////////////////////////////////////////////////////////////////////////
// ForeachStmt

ForeachStmt::ForeachStmt(const std::vector<Symbol *> &lvs, const std::vector<Expr *> &se, const std::vector<Expr *> &ee,
                         Stmt *s, bool t, SourcePos pos)
    : Stmt(pos, ForeachStmtID), dimVariables(lvs), startExprs(se), endExprs(ee), isTiled(t), stmts(s) {}

/* Calculate delta that should be added to varying counter
   between iterations for given dimension.
*/
static llvm::Constant *lCalculateDeltaForVaryingCounter(int dim, int nDims, const std::vector<int> &spans) {
    // Figure out the offsets; this is a little bit tricky.  As an example,
    // consider a 2D tiled foreach loop, where we're running 8-wide and
    // where the inner dimension has a stride of 4 and the outer dimension
    // has a stride of 2.  For the inner dimension, we want the offsets
    // (0,1,2,3,0,1,2,3), and for the outer dimension we want
    // (0,0,0,0,1,1,1,1).
    int32_t delta[ISPC_MAX_NVEC];
    for (int i = 0; i < g->target->getVectorWidth(); ++i) {
        int d = i;
        // First, account for the effect of any dimensions at deeper
        // nesting levels than the current one.
        int prevDimSpanCount = 1;
        for (int j = dim; j < nDims - 1; ++j)
            prevDimSpanCount *= spans[j + 1];

        d /= prevDimSpanCount;
        // And now with what's left, figure out our own offset
        delta[i] = d % spans[dim];
    }

    return LLVMInt32Vector(delta);
}

/* Given a uniform counter value in the memory location pointed to by
   uniformCounterPtr, compute the corresponding set of varying counter
   values for use within the loop body.
 */
static llvm::Value *lUpdateVaryingCounter(int dim, int nDims, FunctionEmitContext *ctx, llvm::Value *uniformCounterPtr,
                                          llvm::Value *varyingCounterPtr, const std::vector<int> &spans) {
    // Smear the uniform counter value out to be varying
    llvm::Value *counter = ctx->LoadInst(uniformCounterPtr);
    llvm::Value *smearCounter = ctx->BroadcastValue(counter, LLVMTypes::Int32VectorType, "smear_counter");

    llvm::Constant *delta = lCalculateDeltaForVaryingCounter(dim, nDims, spans);

    // Add the deltas to compute the varying counter values; store the
    // result to memory and then return it directly as well.
    llvm::Value *varyingCounter = ctx->BinaryOperator(llvm::Instruction::Add, smearCounter, delta, "iter_val");
    ctx->StoreInst(varyingCounter, varyingCounterPtr);
    return varyingCounter;
}

/** Returns the integer log2 of the given integer. */
static int lLog2(int i) {
    int ret = 0;
    while (i != 0) {
        ++ret;
        i >>= 1;
    }
    return ret - 1;
}

/* Figure out how many elements to process in each dimension for each time
   through a foreach loop.  The untiled case is easy; all of the outer
   dimensions up until the innermost one have a span of 1, and the
   innermost one takes the entire vector width.  For the tiled case, we
   give wider spans to the innermost dimensions while also trying to
   generate relatively square domains.

   This code works recursively from outer dimensions to inner dimensions.
 */
static void lGetSpans(int dimsLeft, int nDims, int itemsLeft, bool isTiled, int *a) {
    if (dimsLeft == 0) {
        // Nothing left to do but give all of the remaining work to the
        // innermost domain.
        *a = itemsLeft;
        return;
    }

    if (isTiled == false || (dimsLeft >= lLog2(itemsLeft)))
        // If we're not tiled, or if there are enough dimensions left that
        // giving this one any more than a span of one would mean that a
        // later dimension would have to have a span of one, give this one
        // a span of one to save the available items for later.
        *a = 1;
    else if (itemsLeft >= 16 && (dimsLeft == 1))
        // Special case to have 4x4 domains for the 2D case when running
        // 16-wide.
        *a = 4;
    else
        // Otherwise give this dimension a span of two.
        *a = 2;

    lGetSpans(dimsLeft - 1, nDims, itemsLeft / *a, isTiled, a + 1);
}

/* Emit code for a foreach statement.  We effectively emit code to run the
   set of n-dimensional nested loops corresponding to the dimensionality of
   the foreach statement along with the extra logic to deal with mismatches
   between the vector width we're compiling to and the number of elements
   to process.
 */
void ForeachStmt::EmitCode(FunctionEmitContext *ctx) const {

#ifdef ISPC_XE_ENABLED
    if (ctx->emitXeHardwareMask()) {
        EmitCodeForXe(ctx);
        return;
    }
#endif

    if (ctx->GetCurrentBasicBlock() == NULL || stmts == NULL)
        return;

    llvm::BasicBlock *bbFullBody = ctx->CreateBasicBlock("foreach_full_body");
    llvm::BasicBlock *bbMaskedBody = ctx->CreateBasicBlock("foreach_masked_body");
    llvm::BasicBlock *bbExit = ctx->CreateBasicBlock("foreach_exit");

    llvm::Value *oldMask = ctx->GetInternalMask();
    llvm::Value *oldFunctionMask = ctx->GetFunctionMask();

    ctx->SetDebugPos(pos);
    ctx->StartScope();

    ctx->SetInternalMask(LLVMMaskAllOn);
    ctx->SetFunctionMask(LLVMMaskAllOn);

    // This should be caught during typechecking
    AssertPos(pos, startExprs.size() == dimVariables.size() && endExprs.size() == dimVariables.size());
    int nDims = (int)dimVariables.size();

    ///////////////////////////////////////////////////////////////////////
    // Setup: compute the number of items we have to work on in each
    // dimension and a number of derived values.
    std::vector<llvm::BasicBlock *> bbReset, bbStep, bbTest;
    std::vector<llvm::Value *> startVals, endVals, uniformCounterPtrs;
    std::vector<llvm::Value *> nExtras, alignedEnd, extrasMaskPtrs;

    std::vector<int> span(nDims, 0);
    lGetSpans(nDims - 1, nDims, g->target->getVectorWidth(), isTiled, &span[0]);

    for (int i = 0; i < nDims; ++i) {
        // Basic blocks that we'll fill in later with the looping logic for
        // this dimension.
        bbReset.push_back(ctx->CreateBasicBlock("foreach_reset"));
        if (i < nDims - 1)
            // stepping for the innermost dimension is handled specially
            bbStep.push_back(ctx->CreateBasicBlock("foreach_step"));
        bbTest.push_back(ctx->CreateBasicBlock("foreach_test"));

        // Start and end value for this loop dimension
        llvm::Value *sv = startExprs[i]->GetValue(ctx);
        llvm::Value *ev = endExprs[i]->GetValue(ctx);
        if (sv == NULL || ev == NULL)
            return;
        startVals.push_back(sv);
        endVals.push_back(ev);

        // nItems = endVal - startVal
        llvm::Value *nItems = ctx->BinaryOperator(llvm::Instruction::Sub, ev, sv, "nitems");

        // nExtras = nItems % (span for this dimension)
        // This gives us the number of extra elements we need to deal with
        // at the end of the loop for this dimension that don't fit cleanly
        // into a vector width.
        nExtras.push_back(ctx->BinaryOperator(llvm::Instruction::SRem, nItems, LLVMInt32(span[i]), "nextras"));

        // alignedEnd = endVal - nExtras
        alignedEnd.push_back(ctx->BinaryOperator(llvm::Instruction::Sub, ev, nExtras[i], "aligned_end"));

        ///////////////////////////////////////////////////////////////////////
        // Each dimension has a loop counter that is a uniform value that
        // goes from startVal to endVal, in steps of the span for this
        // dimension.  Its value is only used internally here for looping
        // logic and isn't directly available in the user's program code.
        uniformCounterPtrs.push_back(ctx->AllocaInst(LLVMTypes::Int32Type, "counter"));
        ctx->StoreInst(startVals[i], uniformCounterPtrs[i]);

        // There is also a varying variable that holds the set of index
        // values for each dimension in the current loop iteration; this is
        // the value that is program-visible.
        dimVariables[i]->storagePtr = ctx->AllocaInst(LLVMTypes::Int32VectorType, dimVariables[i]->name.c_str());
        dimVariables[i]->parentFunction = ctx->GetFunction();
        ctx->EmitVariableDebugInfo(dimVariables[i]);

        // Each dimension also maintains a mask that represents which of
        // the varying elements in the current iteration should be
        // processed.  (i.e. this is used to disable the lanes that have
        // out-of-bounds offsets.)
        extrasMaskPtrs.push_back(ctx->AllocaInst(LLVMTypes::MaskType, "extras mask"));
        ctx->StoreInst(LLVMMaskAllOn, extrasMaskPtrs[i]);
    }

    ctx->StartForeach(FunctionEmitContext::FOREACH_REGULAR);

    // On to the outermost loop's test
    ctx->BranchInst(bbTest[0]);

    ///////////////////////////////////////////////////////////////////////////
    // foreach_reset: this code runs when we need to reset the counter for
    // a given dimension in preparation for running through its loop again,
    // after the enclosing level advances its counter.
    for (int i = 0; i < nDims; ++i) {
        ctx->SetCurrentBasicBlock(bbReset[i]);
        if (i == 0)
            ctx->BranchInst(bbExit);
        else {
            ctx->StoreInst(LLVMMaskAllOn, extrasMaskPtrs[i]);
            ctx->StoreInst(startVals[i], uniformCounterPtrs[i]);
            ctx->BranchInst(bbStep[i - 1]);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // foreach_step: increment the uniform counter by the vector width.
    // Note that we don't increment the varying counter here as well but
    // just generate its value when we need it in the loop body.  Don't do
    // this for the innermost dimension, which has a more complex stepping
    // structure..
    for (int i = 0; i < nDims - 1; ++i) {
        ctx->SetCurrentBasicBlock(bbStep[i]);
        llvm::Value *counter = ctx->LoadInst(uniformCounterPtrs[i]);
        llvm::Value *newCounter =
            ctx->BinaryOperator(llvm::Instruction::Add, counter, LLVMInt32(span[i]), "new_counter");
        ctx->StoreInst(newCounter, uniformCounterPtrs[i]);
        ctx->BranchInst(bbTest[i]);
    }

    ///////////////////////////////////////////////////////////////////////////
    // foreach_test (for all dimensions other than the innermost...)
    std::vector<llvm::Value *> inExtras;
    for (int i = 0; i < nDims - 1; ++i) {
        ctx->SetCurrentBasicBlock(bbTest[i]);

        llvm::Value *haveExtras =
            ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SGT, endVals[i], alignedEnd[i], "have_extras");

        llvm::Value *counter = ctx->LoadInst(uniformCounterPtrs[i], NULL, "counter");
        llvm::Value *atAlignedEnd =
            ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, counter, alignedEnd[i], "at_aligned_end");
        llvm::Value *inEx = ctx->BinaryOperator(llvm::Instruction::And, haveExtras, atAlignedEnd, "in_extras");

        if (i == 0)
            inExtras.push_back(inEx);
        else
            inExtras.push_back(ctx->BinaryOperator(llvm::Instruction::Or, inEx, inExtras[i - 1], "in_extras_all"));

        llvm::Value *varyingCounter =
            lUpdateVaryingCounter(i, nDims, ctx, uniformCounterPtrs[i], dimVariables[i]->storagePtr, span);

        llvm::Value *smearEnd = ctx->BroadcastValue(endVals[i], LLVMTypes::Int32VectorType, "smear_end");

        // Do a vector compare of its value to the end value to generate a
        // mask for this last bit of work.
        llvm::Value *emask = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SLT, varyingCounter, smearEnd);
        emask = ctx->I1VecToBoolVec(emask);

        if (i == 0)
            ctx->StoreInst(emask, extrasMaskPtrs[i]);
        else {
            llvm::Value *oldMask = ctx->LoadInst(extrasMaskPtrs[i - 1]);
            llvm::Value *newMask = ctx->BinaryOperator(llvm::Instruction::And, oldMask, emask, "extras_mask");
            ctx->StoreInst(newMask, extrasMaskPtrs[i]);
        }

        llvm::Value *notAtEnd = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SLT, counter, endVals[i]);
        ctx->BranchInst(bbTest[i + 1], bbReset[i], notAtEnd);
    }

    ///////////////////////////////////////////////////////////////////////////
    // foreach_test (for innermost dimension)
    //
    // All of the outer dimensions are handled generically--basically as a
    // for() loop from the start value to the end value, where at each loop
    // test, we compute the mask of active elements for the current
    // dimension and then update an overall mask that is the AND
    // combination of all of the outer ones.
    //
    // The innermost loop is handled specially, for performance purposes.
    // When starting the innermost dimension, we start by checking once
    // whether any of the outer dimensions has set the mask to be
    // partially-active or not.  We follow different code paths for these
    // two cases, taking advantage of the knowledge that the mask is all
    // on, when this is the case.
    //
    // In each of these code paths, we start with a loop from the starting
    // value to the aligned end value for the innermost dimension; we can
    // guarantee that the innermost loop will have an "all on" mask (as far
    // as its dimension is concerned) for the duration of this loop.  Doing
    // so allows us to emit code that assumes the mask is all on (for the
    // case where none of the outer dimensions has set the mask to be
    // partially on), or allows us to emit code that just uses the mask
    // from the outer dimensions directly (for the case where they have).
    //
    // After this loop, we just need to deal with one vector's worth of
    // "ragged extra bits", where the mask used includes the effect of the
    // mask for the innermost dimension.
    //
    // We start out this process by emitting the check that determines
    // whether any of the enclosing dimensions is partially active
    // (i.e. processing extra elements that don't exactly fit into a
    // vector).
    llvm::BasicBlock *bbOuterInExtras = ctx->CreateBasicBlock("outer_in_extras");
    llvm::BasicBlock *bbOuterNotInExtras = ctx->CreateBasicBlock("outer_not_in_extras");

    ctx->SetCurrentBasicBlock(bbTest[nDims - 1]);
    if (inExtras.size()) {
        ctx->BranchInst(bbOuterInExtras, bbOuterNotInExtras, inExtras.back());
    }

    else
        // for a 1D iteration domain, we certainly don't have any enclosing
        // dimensions that are processing extra elements.
        ctx->BranchInst(bbOuterNotInExtras);

    ///////////////////////////////////////////////////////////////////////////
    // One or more outer dimensions in extras, so we need to mask for the loop
    // body regardless.  We break this into two cases, roughly:
    // for (counter = start; counter < alignedEnd; counter += step) {
    //   // mask is all on for inner, so set mask to outer mask
    //   // run loop body with mask
    // }
    // // counter == alignedEnd
    // if (counter < end) {
    //   // set mask to outermask & (counter+programCounter < end)
    //   // run loop body with mask
    // }
    llvm::BasicBlock *bbAllInnerPartialOuter = ctx->CreateBasicBlock("all_inner_partial_outer");
    llvm::BasicBlock *bbPartial = ctx->CreateBasicBlock("both_partial");
    ctx->SetCurrentBasicBlock(bbOuterInExtras);
    {
        // Update the varying counter value here, since all subsequent
        // blocks along this path need it.
        lUpdateVaryingCounter(nDims - 1, nDims, ctx, uniformCounterPtrs[nDims - 1], dimVariables[nDims - 1]->storagePtr,
                              span);

        // here we just check to see if counter < alignedEnd
        llvm::Value *counter = ctx->LoadInst(uniformCounterPtrs[nDims - 1], NULL, "counter");
        llvm::Value *beforeAlignedEnd = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SLT, counter,
                                                     alignedEnd[nDims - 1], "before_aligned_end");
        ctx->BranchInst(bbAllInnerPartialOuter, bbPartial, beforeAlignedEnd);
    }

    // Below we have a basic block that runs the loop body code for the
    // case where the mask is partially but not fully on.  This same block
    // runs in multiple cases: both for handling any ragged extra data for
    // the innermost dimension but also when outer dimensions have set the
    // mask to be partially on.
    //
    // The value stored in stepIndexAfterMaskedBodyPtr is used after each
    // execution of the body code to determine whether the innermost index
    // value should be incremented by the step (we're running the "for"
    // loop of full vectors at the innermost dimension, with outer
    // dimensions having set the mask to be partially on), or whether we're
    // running once for the ragged extra bits at the end of the innermost
    // dimension, in which case we're done with the innermost dimension and
    // should step the loop counter for the next enclosing dimension
    // instead.
    // Revisit : Should this be an i1.
    llvm::Value *stepIndexAfterMaskedBodyPtr = ctx->AllocaInst(LLVMTypes::BoolType, "step_index");

    ///////////////////////////////////////////////////////////////////////////
    // We're in the inner loop part where the only masking is due to outer
    // dimensions but the innermost dimension fits fully into a vector's
    // width.  Set the mask and jump to the masked loop body.
    ctx->SetCurrentBasicBlock(bbAllInnerPartialOuter);
    {
        llvm::Value *mask;
        if (nDims == 1)
            // 1D loop; we shouldn't ever get here anyway
            mask = LLVMMaskAllOff;
        else
            mask = ctx->LoadInst(extrasMaskPtrs[nDims - 2]);

        ctx->SetInternalMask(mask);

        ctx->StoreInst(LLVMTrue, stepIndexAfterMaskedBodyPtr);
        ctx->BranchInst(bbMaskedBody);
    }

    ///////////////////////////////////////////////////////////////////////////
    // We need to include the effect of the innermost dimension in the mask
    // for the final bits here
    ctx->SetCurrentBasicBlock(bbPartial);
    {
        llvm::Value *varyingCounter = ctx->LoadInst(dimVariables[nDims - 1]->storagePtr, dimVariables[nDims - 1]->type);
        llvm::Value *smearEnd = ctx->BroadcastValue(endVals[nDims - 1], LLVMTypes::Int32VectorType, "smear_end");

        llvm::Value *emask = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SLT, varyingCounter, smearEnd);
        emask = ctx->I1VecToBoolVec(emask);

        if (nDims == 1) {
            ctx->SetInternalMask(emask);
        } else {
            llvm::Value *oldMask = ctx->LoadInst(extrasMaskPtrs[nDims - 2]);
            llvm::Value *newMask = ctx->BinaryOperator(llvm::Instruction::And, oldMask, emask, "extras_mask");
            ctx->SetInternalMask(newMask);
        }

        ctx->StoreInst(LLVMFalse, stepIndexAfterMaskedBodyPtr);

        // check to see if counter != end, otherwise, the next step is not necessary
        llvm::Value *counter = ctx->LoadInst(uniformCounterPtrs[nDims - 1], NULL, "counter");
        llvm::Value *atEnd =
            ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE, counter, endVals[nDims - 1], "at_end");
        ctx->BranchInst(bbMaskedBody, bbReset[nDims - 1], atEnd);
    }

    ///////////////////////////////////////////////////////////////////////////
    // None of the outer dimensions is processing extras; along the lines
    // of above, we can express this as:
    // for (counter = start; counter < alignedEnd; counter += step) {
    //   // mask is all on
    //   // run loop body with mask all on
    // }
    // // counter == alignedEnd
    // if (counter < end) {
    //   // set mask to (counter+programCounter < end)
    //   // run loop body with mask
    // }
    llvm::BasicBlock *bbPartialInnerAllOuter = ctx->CreateBasicBlock("partial_inner_all_outer");
    ctx->SetCurrentBasicBlock(bbOuterNotInExtras);
    {
        llvm::Value *counter = ctx->LoadInst(uniformCounterPtrs[nDims - 1], NULL, "counter");
        llvm::Value *beforeAlignedEnd = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SLT, counter,
                                                     alignedEnd[nDims - 1], "before_aligned_end");
        ctx->BranchInst(bbFullBody, bbPartialInnerAllOuter, beforeAlignedEnd);
    }

    ///////////////////////////////////////////////////////////////////////////
    // full_body: do a full vector's worth of work.  We know that all
    // lanes will be running here, so we explicitly set the mask to be 'all
    // on'.  This ends up being relatively straightforward: just update the
    // value of the varying loop counter and have the statements in the
    // loop body emit their code.
    llvm::BasicBlock *bbFullBodyContinue = ctx->CreateBasicBlock("foreach_full_continue");
    ctx->SetCurrentBasicBlock(bbFullBody);
    {
        ctx->SetInternalMask(LLVMMaskAllOn);
        ctx->SetBlockEntryMask(LLVMMaskAllOn);
        lUpdateVaryingCounter(nDims - 1, nDims, ctx, uniformCounterPtrs[nDims - 1], dimVariables[nDims - 1]->storagePtr,
                              span);
        ctx->SetContinueTarget(bbFullBodyContinue);
        ctx->AddInstrumentationPoint("foreach loop body (all on)");
        stmts->EmitCode(ctx);
        AssertPos(pos, ctx->GetCurrentBasicBlock() != NULL);
        ctx->BranchInst(bbFullBodyContinue);
    }
    ctx->SetCurrentBasicBlock(bbFullBodyContinue);
    {
        ctx->RestoreContinuedLanes();
        llvm::Value *counter = ctx->LoadInst(uniformCounterPtrs[nDims - 1]);
        llvm::Value *newCounter =
            ctx->BinaryOperator(llvm::Instruction::Add, counter, LLVMInt32(span[nDims - 1]), "new_counter");
        ctx->StoreInst(newCounter, uniformCounterPtrs[nDims - 1]);
        ctx->BranchInst(bbOuterNotInExtras);
    }

    ///////////////////////////////////////////////////////////////////////////
    // We're done running blocks with the mask all on; see if the counter is
    // less than the end value, in which case we need to run the body one
    // more time to get the extra bits.
    llvm::BasicBlock *bbSetInnerMask = ctx->CreateBasicBlock("partial_inner_only");
    ctx->SetCurrentBasicBlock(bbPartialInnerAllOuter);
    {
        llvm::Value *counter = ctx->LoadInst(uniformCounterPtrs[nDims - 1], NULL, "counter");
        llvm::Value *beforeFullEnd = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SLT, counter,
                                                  endVals[nDims - 1], "before_full_end");
        ctx->BranchInst(bbSetInnerMask, bbReset[nDims - 1], beforeFullEnd);
    }

    ///////////////////////////////////////////////////////////////////////////
    // The outer dimensions are all on, so the mask is just given by the
    // mask for the innermost dimension
    ctx->SetCurrentBasicBlock(bbSetInnerMask);
    {
        llvm::Value *varyingCounter = lUpdateVaryingCounter(nDims - 1, nDims, ctx, uniformCounterPtrs[nDims - 1],
                                                            dimVariables[nDims - 1]->storagePtr, span);
        llvm::Value *smearEnd = ctx->BroadcastValue(endVals[nDims - 1], LLVMTypes::Int32VectorType, "smear_end");
        llvm::Value *emask = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SLT, varyingCounter, smearEnd);
        emask = ctx->I1VecToBoolVec(emask);
        ctx->SetInternalMask(emask);
        ctx->SetBlockEntryMask(emask);

        ctx->StoreInst(LLVMFalse, stepIndexAfterMaskedBodyPtr);
        ctx->BranchInst(bbMaskedBody);
    }

    ///////////////////////////////////////////////////////////////////////////
    // masked_body: set the mask and have the statements emit their
    // code again.  Note that it's generally worthwhile having two copies
    // of the statements' code, since the code above is emitted with the
    // mask known to be all-on, which in turn leads to more efficient code
    // for that case.
    llvm::BasicBlock *bbStepInnerIndex = ctx->CreateBasicBlock("step_inner_index");
    llvm::BasicBlock *bbMaskedBodyContinue = ctx->CreateBasicBlock("foreach_masked_continue");
    ctx->SetCurrentBasicBlock(bbMaskedBody);
    {
        ctx->AddInstrumentationPoint("foreach loop body (masked)");
        ctx->SetContinueTarget(bbMaskedBodyContinue);
        ctx->DisableGatherScatterWarnings();
        ctx->SetBlockEntryMask(ctx->GetFullMask());
        stmts->EmitCode(ctx);
        ctx->EnableGatherScatterWarnings();
        ctx->BranchInst(bbMaskedBodyContinue);
    }
    ctx->SetCurrentBasicBlock(bbMaskedBodyContinue);
    {
        ctx->RestoreContinuedLanes();
        llvm::Value *stepIndex = ctx->LoadInst(stepIndexAfterMaskedBodyPtr);
        ctx->BranchInst(bbStepInnerIndex, bbReset[nDims - 1], stepIndex);
    }

    ///////////////////////////////////////////////////////////////////////////
    // step the innermost index, for the case where we're doing the
    // innermost for loop over full vectors.
    ctx->SetCurrentBasicBlock(bbStepInnerIndex);
    {
        llvm::Value *counter = ctx->LoadInst(uniformCounterPtrs[nDims - 1]);
        llvm::Value *newCounter =
            ctx->BinaryOperator(llvm::Instruction::Add, counter, LLVMInt32(span[nDims - 1]), "new_counter");
        ctx->StoreInst(newCounter, uniformCounterPtrs[nDims - 1]);
        ctx->BranchInst(bbOuterInExtras);
    }

    ///////////////////////////////////////////////////////////////////////////
    // foreach_exit: All done.  Restore the old mask and clean up
    ctx->SetCurrentBasicBlock(bbExit);

    ctx->SetInternalMask(oldMask);
    ctx->SetFunctionMask(oldFunctionMask);

    ctx->EndForeach();
    ctx->EndScope();
}

#ifdef ISPC_XE_ENABLED
/* Emit code for a foreach statement on Xe. We effectively emit code to run
   the set of n-dimensional nested loops corresponding to the dimensionality of
   the foreach statement along with the extra logic to deal with mismatches
   between the vector width we're compiling to and the number of elements
   to process. Handler logic is different from the other targets due to
   Xe Execution Mask usage. We do not need to generate different bodies
   for full and partial masks due to it.
*/
void ForeachStmt::EmitCodeForXe(FunctionEmitContext *ctx) const {
    AssertPos(pos, g->target->isXeTarget());

    if (ctx->GetCurrentBasicBlock() == NULL || stmts == NULL)
        return;

    // We store current EM and reset it to AllOn state.
    llvm::Value *oldMask = ctx->GetInternalMask();
    llvm::Value *oldFunctionMask = ctx->GetFunctionMask();

    llvm::Value *execMask = NULL;
    if (g->opt.enableForeachInsideVarying) {
        Warning(pos, "\"foreach\" statement is not optimized for Xe targets yet.");
        ctx->SetInternalMask(LLVMMaskAllOn);
        ctx->SetFunctionMask(LLVMMaskAllOn);
        execMask = ctx->XeStartUnmaskedRegion();
    } else {
        Warning(pos, "\"foreach\" statement is not supported under varying CF for Xe targets yet. Make sure that"
                     " it is not called under varying CF or use \"--opt=enable-xe-foreach-varying\" to enable its "
                     "experimental support.");
    }
    llvm::BasicBlock *bbBody = ctx->CreateBasicBlock("foreach_body", ctx->GetCurrentBasicBlock());
    llvm::BasicBlock *bbExit = ctx->CreateBasicBlock("foreach_exit", bbBody);

    ctx->SetDebugPos(pos);
    ctx->StartScope();

    // This should be caught during typechecking
    AssertPos(pos, startExprs.size() == dimVariables.size() && endExprs.size() == dimVariables.size());
    int nDims = (int)dimVariables.size();

    ///////////////////////////////////////////////////////////////////////
    // Setup: compute the number of items we have to work on in each
    // dimension and a number of derived values.
    std::vector<llvm::BasicBlock *> bbReset, bbTest, bbStep;
    std::vector<llvm::Value *> startVals, endVals;
    std::vector<llvm::Constant *> steps;
    std::vector<int> span(nDims, 0);
    lGetSpans(nDims - 1, nDims, g->target->getVectorWidth(), isTiled, &span[0]);
    for (int i = 0; i < nDims; ++i) {
        // Basic blocks that we'll fill in later with the looping logic for
        // this dimension.
        bbTest.push_back(ctx->CreateBasicBlock("foreach_test", i == 0 ? ctx->GetCurrentBasicBlock() : bbTest[i - 1]));
        bbStep.push_back(ctx->CreateBasicBlock("foreach_step", bbBody));
        bbReset.push_back(ctx->CreateBasicBlock("foreach_reset", bbStep[i]));

        llvm::Value *sv = startExprs[i]->GetValue(ctx);
        llvm::Value *ev = endExprs[i]->GetValue(ctx);
        if (sv == NULL || ev == NULL)
            return;

        // Store varying start
        sv = ctx->BroadcastValue(sv, LLVMTypes::Int32VectorType, "start_broadcast");
        llvm::Constant *delta = lCalculateDeltaForVaryingCounter(i, nDims, span);
        sv = ctx->BinaryOperator(llvm::Instruction::Add, sv, delta, "varying_start");
        startVals.push_back(sv);

        // Store broadcasted end values
        ev = ctx->BroadcastValue(ev, LLVMTypes::Int32VectorType, "end_broadcast");
        endVals.push_back(ev);

        // Store vectorized step
        llvm::Constant *step = LLVMInt32Vector(span[i]);
        steps.push_back(step);

        // Init vectorized counters
        dimVariables[i]->storagePtr = ctx->AllocaInst(LLVMTypes::Int32VectorType, dimVariables[i]->name.c_str());
        dimVariables[i]->parentFunction = ctx->GetFunction();
        ctx->StoreInst(sv, dimVariables[i]->storagePtr);
        ctx->EmitVariableDebugInfo(dimVariables[i]);
    }

    // Officially start foreach. Emulating uniform for proper continue handlers.
    ctx->StartForeach(FunctionEmitContext::FOREACH_REGULAR, true);

    // Jump to outermost test block
    ctx->BranchInst(bbTest[0]);

    ///////////////////////////////////////////////////////////////////////////
    // foreach_reset: this code runs when we need to reset the counter for
    // a given dimension in preparation for running through its loop again,
    // after the enclosing level advances its counter.
    for (int i = 0; i < nDims; ++i) {
        ctx->SetCurrentBasicBlock(bbReset[i]);
        if (i == 0)
            // Outermost loop finished - exit
            ctx->BranchInst(bbExit);
        else {
            // Reset counter for this dimension, iterate over previous one
            ctx->StoreInst(startVals[i], dimVariables[i]->storagePtr);
            ctx->BranchInst(bbStep[i - 1]);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // foreach_step: iterate counters with step that was calculated before
    // entering foreach.
    for (int i = 0; i < nDims; ++i) {
        ctx->SetCurrentBasicBlock(bbStep[i]);
        llvm::Value *counter = ctx->LoadInst(dimVariables[i]->storagePtr);
        llvm::Value *newCounter = ctx->BinaryOperator(llvm::Instruction::Add, counter, steps[i], "new_counter");
        ctx->StoreInst(newCounter, dimVariables[i]->storagePtr);
        ctx->BranchInst(bbTest[i]);
    }

    ///////////////////////////////////////////////////////////////////////////
    // foreach_test: compare varying counter with end value and branch to
    // target or reset. Xe EM magic happens here: we turn off all lanes
    // that fail check until reset is reached. And reset is reached only when
    // all lanes fail this check due to test -> target -> step -> test loop.
    //
    // It looks tricky for multidimensional case. Suppose we have 3 dimensional
    // loop, some lanes were turn off in the second dimension. When we reach
    // reset in the innermost one (3rd) we won't be able to reset lanes that were
    // turned off in the second dimension. But we don't actually need to reset
    // them: they were reseted right before test of the second dimension turned
    // them off. So after 2nd dimension's reset there will be reseted 2nd and 3rd
    // counters. If some of lanes were turned off in the first dimension all
    // this stuff doesn't matter: we will exit loop after this iteration anyway.
    for (int i = 0; i < nDims; ++i) {
        ctx->SetCurrentBasicBlock(bbTest[i]);
        llvm::Value *val = ctx->LoadInst(dimVariables[i]->storagePtr, NULL, "val");
        llvm::Value *checkVal = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SLT, val, endVals[i]);
        // Target is body for innermost dimension, next dimension test for others
        llvm::BasicBlock *targetBB = (i < nDims - 1) ? bbTest[i + 1] : bbBody;
        // Turn off lanes untill reset is reached
        ctx->BranchInst(targetBB, bbReset[i], checkVal);
    }

    ///////////////////////////////////////////////////////////////////////////
    // foreach_body: emit code for loop body. Execution is driven by
    // Xe Execution Mask.
    ctx->SetCurrentBasicBlock(bbBody);
    ctx->SetContinueTarget(bbStep[nDims - 1]);
    ctx->AddInstrumentationPoint("foreach loop body");
    stmts->EmitCode(ctx);
    AssertPos(pos, ctx->GetCurrentBasicBlock() != NULL);
    ctx->BranchInst(bbStep[nDims - 1]);

    ///////////////////////////////////////////////////////////////////////////
    // foreach_exit: All done. Restore the old mask and clean up
    ctx->SetCurrentBasicBlock(bbExit);

    // Restore execution mask from value that was saved at the beginning
    if (execMask != NULL) {
        ctx->XeEndUnmaskedRegion(execMask);
        ctx->SetInternalMask(oldMask);
        ctx->SetFunctionMask(oldFunctionMask);
    }

    ctx->EndForeach();
    ctx->EndScope();
}
#endif

Stmt *ForeachStmt::TypeCheck() {
    bool anyErrors = false;
    for (unsigned int i = 0; i < startExprs.size(); ++i) {
        if (startExprs[i] != NULL)
            startExprs[i] = TypeConvertExpr(startExprs[i], AtomicType::UniformInt32, "foreach starting value");
        anyErrors |= (startExprs[i] == NULL);
    }
    for (unsigned int i = 0; i < endExprs.size(); ++i) {
        if (endExprs[i] != NULL)
            endExprs[i] = TypeConvertExpr(endExprs[i], AtomicType::UniformInt32, "foreach ending value");
        anyErrors |= (endExprs[i] == NULL);
    }

    if (startExprs.size() < dimVariables.size()) {
        Error(pos,
              "Not enough initial values provided for \"foreach\" loop; "
              "got %d, expected %d\n",
              (int)startExprs.size(), (int)dimVariables.size());
        anyErrors = true;
    } else if (startExprs.size() > dimVariables.size()) {
        Error(pos,
              "Too many initial values provided for \"foreach\" loop; "
              "got %d, expected %d\n",
              (int)startExprs.size(), (int)dimVariables.size());
        anyErrors = true;
    }

    if (endExprs.size() < dimVariables.size()) {
        Error(pos,
              "Not enough initial values provided for \"foreach\" loop; "
              "got %d, expected %d\n",
              (int)endExprs.size(), (int)dimVariables.size());
        anyErrors = true;
    } else if (endExprs.size() > dimVariables.size()) {
        Error(pos,
              "Too many initial values provided for \"foreach\" loop; "
              "got %d, expected %d\n",
              (int)endExprs.size(), (int)dimVariables.size());
        anyErrors = true;
    }

    return anyErrors ? NULL : this;
}

void ForeachStmt::SetLoopAttribute(std::pair<Globals::pragmaUnrollType, int> lAttr) {
    Warning(pos, "'#pragma unroll/nounroll' ignored - not supported for foreach loop.");
}

int ForeachStmt::EstimateCost() const { return dimVariables.size() * (COST_UNIFORM_LOOP + COST_SIMPLE_ARITH_LOGIC_OP); }

void ForeachStmt::Print(int indent) const {
    printf("%*cForeach Stmt", indent, ' ');
    pos.Print();
    printf("\n");

    for (unsigned int i = 0; i < dimVariables.size(); ++i)
        if (dimVariables[i] != NULL)
            printf("%*cVar %d: %s\n", indent + 4, ' ', i, dimVariables[i]->name.c_str());
        else
            printf("%*cVar %d: NULL\n", indent + 4, ' ', i);

    printf("Start values:\n");
    for (unsigned int i = 0; i < startExprs.size(); ++i) {
        if (startExprs[i] != NULL)
            startExprs[i]->Print();
        else
            printf("NULL");
        if (i != startExprs.size() - 1)
            printf(", ");
        else
            printf("\n");
    }

    printf("End values:\n");
    for (unsigned int i = 0; i < endExprs.size(); ++i) {
        if (endExprs[i] != NULL)
            endExprs[i]->Print();
        else
            printf("NULL");
        if (i != endExprs.size() - 1)
            printf(", ");
        else
            printf("\n");
    }

    if (stmts != NULL) {
        printf("%*cStmts:\n", indent + 4, ' ');
        stmts->Print(indent + 8);
    }
}

///////////////////////////////////////////////////////////////////////////
// ForeachActiveStmt

ForeachActiveStmt::ForeachActiveStmt(Symbol *s, Stmt *st, SourcePos pos) : Stmt(pos, ForeachActiveStmtID) {
    sym = s;
    stmts = st;
}

void ForeachActiveStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock())
        return;

    // Allocate storage for the symbol that we'll use for the uniform
    // variable that holds the current program instance in each loop
    // iteration.
    if (sym->type == NULL) {
        Assert(m->errorCount > 0);
        return;
    }
    Assert(Type::Equal(sym->type, AtomicType::UniformInt64->GetAsConstType()));
    sym->storagePtr = ctx->AllocaInst(LLVMTypes::Int64Type, sym->name.c_str());

    ctx->SetDebugPos(pos);
    ctx->EmitVariableDebugInfo(sym);

    // The various basic blocks that we'll need in the below
    llvm::BasicBlock *bbFindNext = ctx->CreateBasicBlock("foreach_active_find_next", ctx->GetCurrentBasicBlock());
    llvm::BasicBlock *bbBody = ctx->CreateBasicBlock("foreach_active_body", bbFindNext);
    llvm::BasicBlock *bbCheckForMore = ctx->CreateBasicBlock("foreach_active_check_for_more", bbBody);
    llvm::BasicBlock *bbDone = ctx->CreateBasicBlock("foreach_active_done", bbCheckForMore);

    // Save the old mask so that we can restore it at the end
    llvm::Value *oldInternalMask = ctx->GetInternalMask();

    // Now, *maskBitsPtr will maintain a bitmask for the lanes that remain
    // to be processed by a pass through the loop body.  It starts out with
    // the current execution mask (which should never be all off going in
    // to this)...
    llvm::Value *oldFullMask = NULL;
    bool uniformEmulated = false;
#ifdef ISPC_XE_ENABLED
    if (ctx->emitXeHardwareMask()) {
        // Emulate uniform to make proper continue handler
        uniformEmulated = true;
        // Current mask will be calculated according to EM mask
        oldFullMask = ctx->XeSimdCFPredicate(LLVMMaskAllOn);
    } else
#endif
        oldFullMask = ctx->GetFullMask();

    llvm::Value *maskBitsPtr = ctx->AllocaInst(LLVMTypes::Int64Type, "mask_bits");
    llvm::Value *movmsk = ctx->LaneMask(oldFullMask);
    ctx->StoreInst(movmsk, maskBitsPtr);

    // Officially start the loop.
    ctx->StartScope();
    ctx->StartForeach(FunctionEmitContext::FOREACH_ACTIVE, uniformEmulated);
    ctx->SetContinueTarget(bbCheckForMore);

    // Onward to find the first set of program instance to run the loop for
    ctx->BranchInst(bbFindNext);

    ctx->SetCurrentBasicBlock(bbFindNext);
    {
        // Load the bitmask of the lanes left to be processed
        llvm::Value *remainingBits = ctx->LoadInst(maskBitsPtr, NULL, "remaining_bits");

        // Find the index of the first set bit in the mask
        llvm::Function *ctlzFunc = m->module->getFunction("__count_trailing_zeros_i64");
        Assert(ctlzFunc != NULL);
        llvm::Value *firstSet = ctx->CallInst(ctlzFunc, NULL, remainingBits, "first_set");

        // Store that value into the storage allocated for the iteration
        // variable.
        ctx->StoreInst(firstSet, sym->storagePtr, sym->type);

        // Now set the execution mask to be only on for the current program
        // instance.  (TODO: is there a more efficient way to do this? e.g.
        // for AVX1, we might want to do this as float rather than int
        // math...)

        // Get the "program index" vector value
        llvm::Value *programIndex = ctx->ProgramIndexVector();

        // And smear the current lane out to a vector
        llvm::Value *firstSet32 = ctx->TruncInst(firstSet, LLVMTypes::Int32Type, "first_set32");
        llvm::Value *firstSet32Smear = ctx->SmearUniform(firstSet32);

        // Now set the execution mask based on doing a vector compare of
        // these two
        llvm::Value *iterMask =
            ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, firstSet32Smear, programIndex);
        iterMask = ctx->I1VecToBoolVec(iterMask);

        // Don't need to change this mask in XE: execution
        // is performed according to Xe EM
        if (!ctx->emitXeHardwareMask())
            ctx->SetInternalMask(iterMask);

        // Also update the bitvector of lanes left to turn off the bit for
        // the lane we're about to run.
        llvm::Value *setMask = ctx->BinaryOperator(llvm::Instruction::Shl, LLVMInt64(1), firstSet, "set_mask");
        llvm::Value *notSetMask = ctx->NotOperator(setMask);
        llvm::Value *newRemaining =
            ctx->BinaryOperator(llvm::Instruction::And, remainingBits, notSetMask, "new_remaining");
        ctx->StoreInst(newRemaining, maskBitsPtr);

        // and onward to run the loop body...
        // Set Xe EM through simdcf.goto
        // The EM will be restored when CheckForMore is reached
        if (ctx->emitXeHardwareMask()) {
            ctx->BranchInst(bbBody, bbCheckForMore, iterMask);
        } else {
            ctx->BranchInst(bbBody);
        }
    }

    ctx->SetCurrentBasicBlock(bbBody);
    {
        ctx->SetBlockEntryMask(ctx->GetFullMask());

        // Run the code in the body of the loop.  This is easy now.
        if (stmts)
            stmts->EmitCode(ctx);

        Assert(ctx->GetCurrentBasicBlock() != NULL);
        ctx->BranchInst(bbCheckForMore);
    }

    ctx->SetCurrentBasicBlock(bbCheckForMore);
    {
        ctx->RestoreContinuedLanes();
        // At the end of the loop body (either due to running the
        // statements normally, or a continue statement in the middle of
        // the loop that jumps to the end, see if there are any lanes left
        // to be processed.
        llvm::Value *remainingBits = ctx->LoadInst(maskBitsPtr, NULL, "remaining_bits");
        llvm::Value *nonZero = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE, remainingBits,
                                            LLVMInt64(0), "remaining_ne_zero");
        ctx->BranchInst(bbFindNext, bbDone, nonZero);
    }

    ctx->SetCurrentBasicBlock(bbDone);
    ctx->SetInternalMask(oldInternalMask);
    ctx->EndForeach();
    ctx->EndScope();
}

void ForeachActiveStmt::Print(int indent) const {
    printf("%*cForeach_active Stmt", indent, ' ');
    pos.Print();
    printf("\n");

    printf("%*cIter symbol: ", indent + 4, ' ');
    if (sym != NULL) {
        printf("%s", sym->name.c_str());
        if (sym->type != NULL)
            printf(" %s", sym->type->GetString().c_str());
    } else
        printf("NULL");
    printf("\n");

    printf("%*cStmts:\n", indent + 4, ' ');
    if (stmts != NULL)
        stmts->Print(indent + 8);
    else
        printf("NULL");
    printf("\n");
}

Stmt *ForeachActiveStmt::TypeCheck() {
    if (sym == NULL)
        return NULL;

    return this;
}

void ForeachActiveStmt::SetLoopAttribute(std::pair<Globals::pragmaUnrollType, int> lAttr) {
    Warning(pos, "'#pragma unroll/nounroll' ignored - not supported for foreach_active loop.");
}

int ForeachActiveStmt::EstimateCost() const { return COST_VARYING_LOOP; }

///////////////////////////////////////////////////////////////////////////
// ForeachUniqueStmt

ForeachUniqueStmt::ForeachUniqueStmt(const char *iterName, Expr *e, Stmt *s, SourcePos pos)
    : Stmt(pos, ForeachUniqueStmtID) {
    sym = m->symbolTable->LookupVariable(iterName);
    expr = e;
    stmts = s;
}

void ForeachUniqueStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock())
        return;

    // First, allocate local storage for the symbol that we'll use for the
    // uniform variable that holds the current unique value through each
    // loop.
    if (sym->type == NULL) {
        Assert(m->errorCount > 0);
        return;
    }
    llvm::Type *symType = sym->type->LLVMType(g->ctx);
    if (symType == NULL) {
        Assert(m->errorCount > 0);
        return;
    }

    sym->storagePtr = ctx->AllocaInst(sym->type, sym->name.c_str());

    ctx->SetDebugPos(pos);
    ctx->EmitVariableDebugInfo(sym);

    // The various basic blocks that we'll need in the below
    llvm::BasicBlock *bbFindNext = ctx->CreateBasicBlock("foreach_find_next", ctx->GetCurrentBasicBlock());
    llvm::BasicBlock *bbBody = ctx->CreateBasicBlock("foreach_body", bbFindNext);
    llvm::BasicBlock *bbCheckForMore = ctx->CreateBasicBlock("foreach_check_for_more", bbBody);
    llvm::BasicBlock *bbDone = ctx->CreateBasicBlock("foreach_done", bbCheckForMore);

    // Prepare the FunctionEmitContext
    ctx->StartScope();

    // Save the old internal mask so that we can restore it at the end
    llvm::Value *oldMask = ctx->GetInternalMask();

    // Now, *maskBitsPtr will maintain a bitmask for the lanes that remain
    // to be processed by a pass through the foreach_unique loop body.  It
    // starts out with the full execution mask (which should never be all
    // off going in to this)...
    llvm::Value *oldFullMask = NULL;
    bool emulatedUniform = false;
#ifdef ISPC_XE_ENABLED
    if (ctx->emitXeHardwareMask()) {
        // Emulating uniform behavior for proper continue handling
        emulatedUniform = true;
        // Current mask will be calculated according to EM mask
        oldFullMask = ctx->XeSimdCFPredicate(LLVMMaskAllOn);
    } else
#endif
        oldFullMask = ctx->GetFullMask();

    llvm::Value *maskBitsPtr = ctx->AllocaInst(LLVMTypes::Int64Type, "mask_bits");
    llvm::Value *movmsk = ctx->LaneMask(oldFullMask);
    ctx->StoreInst(movmsk, maskBitsPtr);

    // Officially start the loop.
    ctx->StartForeach(FunctionEmitContext::FOREACH_UNIQUE, emulatedUniform);
    ctx->SetContinueTarget(bbCheckForMore);

    // Evaluate the varying expression we're iterating over just once.
    llvm::Value *exprValue = expr->GetValue(ctx);

    // And we'll store its value into locally-allocated storage, for ease
    // of indexing over it with non-compile-time-constant indices.
    const Type *exprType;
    if (exprValue == NULL || (exprType = expr->GetType()) == NULL ||
        llvm::dyn_cast<llvm::VectorType>(exprValue->getType()) == NULL) {
        Assert(m->errorCount > 0);
        return;
    }
    ctx->SetDebugPos(pos);
    const Type *exprPtrType = PointerType::GetUniform(exprType);
    llvm::Value *exprMem = ctx->AllocaInst(exprType, "expr_mem");
    ctx->StoreInst(exprValue, exprMem, exprType);

    // Onward to find the first set of lanes to run the loop for
    ctx->BranchInst(bbFindNext);

    ctx->SetCurrentBasicBlock(bbFindNext);
    {
        // Load the bitmask of the lanes left to be processed
        llvm::Value *remainingBits = ctx->LoadInst(maskBitsPtr, NULL, "remaining_bits");

        // Find the index of the first set bit in the mask
        llvm::Function *ctlzFunc = m->module->getFunction("__count_trailing_zeros_i64");
        Assert(ctlzFunc != NULL);
        llvm::Value *firstSet = ctx->CallInst(ctlzFunc, NULL, remainingBits, "first_set");

        // And load the corresponding element value from the temporary
        // memory storing the value of the varying expr.
        llvm::Value *uniqueValue;
        llvm::Value *uniqueValuePtr =
            ctx->GetElementPtrInst(exprMem, LLVMInt64(0), firstSet, exprPtrType, "unique_index_ptr");
        uniqueValue = ctx->LoadInst(uniqueValuePtr, exprType, "unique_value");
        // If it's a varying pointer type, need to convert from the int
        // type we store in the vector to the actual pointer type
        if (llvm::dyn_cast<llvm::PointerType>(symType) != NULL)
            uniqueValue = ctx->IntToPtrInst(uniqueValue, symType);
        Assert(uniqueValue != NULL);
        // Store that value in sym's storage so that the iteration variable
        // has the right value inside the loop body
        ctx->StoreInst(uniqueValue, sym->storagePtr, sym->type);

        // Set the execution mask so that it's on for any lane that a) was
        // running at the start of the foreach loop, and b) where that
        // lane's value of the varying expression is the same as the value
        // we've selected to process this time through--i.e.:
        // oldMask & (smear(element) == exprValue)
        llvm::Value *uniqueSmear = ctx->SmearUniform(uniqueValue, "unique_smear");
        llvm::Value *matchingLanes = NULL;
        if (uniqueValue->getType()->isFloatingPointTy())
            matchingLanes = ctx->CmpInst(llvm::Instruction::FCmp, llvm::CmpInst::FCMP_OEQ, uniqueSmear, exprValue,
                                         "matching_lanes");
        else
            matchingLanes =
                ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, uniqueSmear, exprValue, "matching_lanes");
        matchingLanes = ctx->I1VecToBoolVec(matchingLanes);

        llvm::Value *loopMask =
            ctx->BinaryOperator(llvm::Instruction::And, oldMask, matchingLanes, "foreach_unique_loop_mask");

        // Don't need to change this mask in XE: execution
        // is performed according to Xe EM
        if (!ctx->emitXeHardwareMask())
            ctx->SetInternalMask(loopMask);

        // Also update the bitvector of lanes left to process in subsequent
        // loop iterations:
        // remainingBits &= ~movmsk(current mask)
        llvm::Value *loopMaskMM = ctx->LaneMask(loopMask);
        llvm::Value *notLoopMaskMM = ctx->NotOperator(loopMaskMM);
        llvm::Value *newRemaining =
            ctx->BinaryOperator(llvm::Instruction::And, remainingBits, notLoopMaskMM, "new_remaining");
        ctx->StoreInst(newRemaining, maskBitsPtr);

        // and onward...
        // Set Xe EM through simdcf.goto
        // The EM will be restored when CheckForMore is reached
        if (ctx->emitXeHardwareMask()) {
            ctx->BranchInst(bbBody, bbCheckForMore, loopMask);
        } else {
            ctx->BranchInst(bbBody);
        }
    }

    ctx->SetCurrentBasicBlock(bbBody);
    {
        ctx->SetBlockEntryMask(ctx->GetFullMask());
        // Run the code in the body of the loop.  This is easy now.
        if (stmts)
            stmts->EmitCode(ctx);

        Assert(ctx->GetCurrentBasicBlock() != NULL);
        ctx->BranchInst(bbCheckForMore);
    }

    ctx->SetCurrentBasicBlock(bbCheckForMore);
    {
        // At the end of the loop body (either due to running the
        // statements normally, or a continue statement in the middle of
        // the loop that jumps to the end, see if there are any lanes left
        // to be processed.
        ctx->RestoreContinuedLanes();
        llvm::Value *remainingBits = ctx->LoadInst(maskBitsPtr, NULL, "remaining_bits");
        llvm::Value *nonZero = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE, remainingBits,
                                            LLVMInt64(0), "remaining_ne_zero");
        ctx->BranchInst(bbFindNext, bbDone, nonZero);
    }

    ctx->SetCurrentBasicBlock(bbDone);
    ctx->SetInternalMask(oldMask);
    ctx->EndForeach();
    ctx->EndScope();
}

void ForeachUniqueStmt::Print(int indent) const {
    printf("%*cForeach_unique Stmt", indent, ' ');
    pos.Print();
    printf("\n");

    printf("%*cIter symbol: ", indent + 4, ' ');
    if (sym != NULL) {
        printf("%s", sym->name.c_str());
        if (sym->type != NULL)
            printf(" %s", sym->type->GetString().c_str());
    } else
        printf("NULL");
    printf("\n");

    printf("%*cIter expr: ", indent + 4, ' ');
    if (expr != NULL)
        expr->Print();
    else
        printf("NULL");
    printf("\n");

    printf("%*cStmts:\n", indent + 4, ' ');
    if (stmts != NULL)
        stmts->Print(indent + 8);
    else
        printf("NULL");
    printf("\n");
}

Stmt *ForeachUniqueStmt::TypeCheck() {
    const Type *type;
    if (sym == NULL || expr == NULL || (type = expr->GetType()) == NULL)
        return NULL;

    if (type->IsVaryingType() == false) {
        Error(expr->pos,
              "Iteration domain type in \"foreach_tiled\" loop "
              "must be \"varying\" type, not \"%s\".",
              type->GetString().c_str());
        return NULL;
    }

    if (Type::IsBasicType(type) == false) {
        Error(expr->pos,
              "Iteration domain type in \"foreach_tiled\" loop "
              "must be an atomic, pointer, or enum type, not \"%s\".",
              type->GetString().c_str());
        return NULL;
    }

    return this;
}

void ForeachUniqueStmt::SetLoopAttribute(std::pair<Globals::pragmaUnrollType, int> lAttr) {
    Warning(pos, "'#pragma unroll/nounroll' ignored - not supported for foreach_unique loop.");
}

int ForeachUniqueStmt::EstimateCost() const { return COST_VARYING_LOOP; }

///////////////////////////////////////////////////////////////////////////
// CaseStmt

/** Given the statements following a 'case' or 'default' label, this
    function determines whether the mask should be checked to see if it is
    "all off" immediately after the label, before executing the code for
    the statements.
 */
static bool lCheckMask(Stmt *stmts) {
    if (stmts == NULL)
        return false;

    int cost = EstimateCost(stmts);
    bool safeToRunWithAllLanesOff = SafeToRunWithMaskAllOff(stmts);

    // The mask should be checked if the code following the
    // 'case'/'default' is relatively complex, or if it would be unsafe to
    // run that code with the execution mask all off.
    return (cost > PREDICATE_SAFE_IF_STATEMENT_COST || safeToRunWithAllLanesOff == false);
}

CaseStmt::CaseStmt(int v, Stmt *s, SourcePos pos) : Stmt(pos, CaseStmtID), value(v) { stmts = s; }

void CaseStmt::EmitCode(FunctionEmitContext *ctx) const {
    ctx->EmitCaseLabel(value, lCheckMask(stmts), pos);
    if (stmts)
        stmts->EmitCode(ctx);
}

void CaseStmt::Print(int indent) const {
    printf("%*cCase [%d] label", indent, ' ', value);
    pos.Print();
    printf("\n");
    stmts->Print(indent + 4);
}

Stmt *CaseStmt::TypeCheck() { return this; }

int CaseStmt::EstimateCost() const { return 0; }

///////////////////////////////////////////////////////////////////////////
// DefaultStmt

DefaultStmt::DefaultStmt(Stmt *s, SourcePos pos) : Stmt(pos, DefaultStmtID) { stmts = s; }

void DefaultStmt::EmitCode(FunctionEmitContext *ctx) const {
    ctx->EmitDefaultLabel(lCheckMask(stmts), pos);
    if (stmts)
        stmts->EmitCode(ctx);
}

void DefaultStmt::Print(int indent) const {
    printf("%*cDefault Stmt", indent, ' ');
    pos.Print();
    printf("\n");
    stmts->Print(indent + 4);
}

Stmt *DefaultStmt::TypeCheck() { return this; }

int DefaultStmt::EstimateCost() const { return 0; }

///////////////////////////////////////////////////////////////////////////
// SwitchStmt

SwitchStmt::SwitchStmt(Expr *e, Stmt *s, SourcePos pos) : Stmt(pos, SwitchStmtID) {
    expr = e;
    stmts = s;
}

/* An instance of this structure is carried along as we traverse the AST
   nodes for the statements after a "switch" statement.  We use this
   structure to record all of the 'case' and 'default' statements after the
   "switch". */
struct SwitchVisitInfo {
    SwitchVisitInfo(FunctionEmitContext *c) {
        ctx = c;
        defaultBlock = NULL;
        lastBlock = NULL;
        insertAfter = ctx->GetCurrentBasicBlock();
    }

    FunctionEmitContext *ctx;

    /* Basic block for the code following the "default" label (if any). */
    llvm::BasicBlock *defaultBlock;

    /* Map from integer values after "case" labels to the basic blocks that
       follow the corresponding "case" label. */
    std::vector<std::pair<int, llvm::BasicBlock *>> caseBlocks;

    /* For each basic block for a "case" label or a "default" label,
       nextBlock[block] stores the basic block pointer for the next
       subsequent "case" or "default" label in the program. */
    std::map<llvm::BasicBlock *, llvm::BasicBlock *> nextBlock;

    /* The last basic block created for a "case" or "default" label; when
       we create the basic block for the next one, we'll use this to update
       the nextBlock map<> above. */
    llvm::BasicBlock *lastBlock;

    llvm::BasicBlock *insertAfter;
};

static bool lSwitchASTPreVisit(ASTNode *node, void *d) {
    if (llvm::dyn_cast<SwitchStmt>(node) != NULL)
        // don't continue recursively into a nested switch--we only want
        // our own case and default statements!
        return false;

    CaseStmt *cs = llvm::dyn_cast<CaseStmt>(node);
    DefaultStmt *ds = llvm::dyn_cast<DefaultStmt>(node);

    SwitchVisitInfo *svi = (SwitchVisitInfo *)d;
    llvm::BasicBlock *bb = NULL;
    if (cs != NULL) {
        // Complain if we've seen a case statement with the same value
        // already
        for (int i = 0; i < (int)svi->caseBlocks.size(); ++i) {
            if (svi->caseBlocks[i].first == cs->value) {
                Error(cs->pos, "Duplicate case value \"%d\".", cs->value);
                return true;
            }
        }

        // Otherwise create a new basic block for the code following this
        // 'case' statement and record the mappign between the case label
        // value and the basic block
        char buf[32];
        snprintf(buf, sizeof(buf), "case_%d", cs->value);
        bb = svi->ctx->CreateBasicBlock(buf, svi->insertAfter);
        svi->caseBlocks.push_back(std::make_pair(cs->value, bb));
    } else if (ds != NULL) {
        // And complain if we've seen another 'default' label..
        if (svi->defaultBlock != NULL) {
            Error(ds->pos, "Multiple \"default\" lables in switch statement.");
            return true;
        } else {
            // Otherwise create a basic block for the code following the
            // "default".
            bb = svi->ctx->CreateBasicBlock("default", svi->insertAfter);
            svi->defaultBlock = bb;
        }
    }

    // If we saw a "case" or "default" label, then update the map to record
    // that the block we just created follows the block created for the
    // previous label in the "switch".
    if (bb != NULL) {
        svi->nextBlock[svi->lastBlock] = bb;
        svi->lastBlock = bb;
        svi->insertAfter = bb;
    }

    return true;
}

void SwitchStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (ctx->GetCurrentBasicBlock() == NULL)
        return;

    const Type *type;
    if (expr == NULL || ((type = expr->GetType()) == NULL)) {
        AssertPos(pos, m->errorCount > 0);
        return;
    }

    // Basic block we'll end up after the switch statement
    llvm::BasicBlock *bbDone = ctx->CreateBasicBlock("switch_done", ctx->GetCurrentBasicBlock());

    // Walk the AST of the statements after the 'switch' to collect a bunch
    // of information about the structure of the 'case' and 'default'
    // statements.
    SwitchVisitInfo svi(ctx);
    WalkAST(stmts, lSwitchASTPreVisit, NULL, &svi);
    // Record that the basic block following the last one created for a
    // case/default is the block after the end of the switch statement.
    svi.nextBlock[svi.lastBlock] = bbDone;

    llvm::Value *exprValue = expr->GetValue(ctx);
    if (exprValue == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return;
    }

    bool isUniformCF = (type->IsUniformType() && lHasVaryingBreakOrContinue(stmts) == false);
    bool emulateUniform = false;
#ifdef ISPC_XE_ENABLED
    if (ctx->emitXeHardwareMask()) {
        if (isUniformCF && ctx->inXeSimdCF()) {
            // Broadcast value to work with EM. We are doing
            // it here because it is too late to make CMP
            // broadcast through BranchInst: we need vectorized
            // case checks to be able to reenable fall through
            // cases under emulated uniform CF.
            llvm::Type *vecType = (exprValue->getType() == LLVMTypes::Int32Type) ? LLVMTypes::Int32VectorType
                                                                                 : LLVMTypes::Int64VectorType;
            exprValue = ctx->BroadcastValue(exprValue, vecType, "switch_expr_broadcast");
            emulateUniform = true;
        }

        if (!isUniformCF) {
            isUniformCF = true;
            emulateUniform = true;
        }
    }
#endif

    ctx->StartSwitch(isUniformCF, bbDone, emulateUniform);
    ctx->SetBlockEntryMask(ctx->GetFullMask());
    ctx->SwitchInst(exprValue, svi.defaultBlock ? svi.defaultBlock : bbDone, svi.caseBlocks, svi.nextBlock);

    if (stmts != NULL)
        stmts->EmitCode(ctx);

    if (ctx->GetCurrentBasicBlock() != NULL)
        ctx->BranchInst(bbDone);

    ctx->SetCurrentBasicBlock(bbDone);
    ctx->EndSwitch();
}

void SwitchStmt::Print(int indent) const {
    printf("%*cSwitch Stmt", indent, ' ');
    pos.Print();
    printf("\n");
    printf("%*cexpr = ", indent, ' ');
    expr->Print();
    printf("\n");
    stmts->Print(indent + 4);
}

Stmt *SwitchStmt::TypeCheck() {
    const Type *exprType;
    if (expr == NULL || (exprType = expr->GetType()) == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    const Type *toType = NULL;
    exprType = exprType->GetAsConstType();
    bool is64bit = (Type::EqualIgnoringConst(exprType->GetAsUniformType(), AtomicType::UniformUInt64) ||
                    Type::EqualIgnoringConst(exprType->GetAsUniformType(), AtomicType::UniformInt64));

    if (exprType->IsUniformType()) {
        if (is64bit)
            toType = AtomicType::UniformInt64;
        else
            toType = AtomicType::UniformInt32;
    } else {
        if (is64bit)
            toType = AtomicType::VaryingInt64;
        else
            toType = AtomicType::VaryingInt32;
    }

    expr = TypeConvertExpr(expr, toType, "switch expression");
    if (expr == NULL)
        return NULL;

    return this;
}

int SwitchStmt::EstimateCost() const {
    const Type *type = expr->GetType();
    if (type && type->IsVaryingType())
        return COST_VARYING_SWITCH;
    else
        return COST_UNIFORM_SWITCH;
}

///////////////////////////////////////////////////////////////////////////
// UnmaskedStmt

UnmaskedStmt::UnmaskedStmt(Stmt *s, SourcePos pos) : Stmt(pos, UnmaskedStmtID) { stmts = s; }

void UnmaskedStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock() || !stmts)
        return;
    llvm::Value *oldInternalMask = ctx->GetInternalMask();
    llvm::Value *oldFunctionMask = ctx->GetFunctionMask();

    ctx->SetInternalMask(LLVMMaskAllOn);
    ctx->SetFunctionMask(LLVMMaskAllOn);
    if (!ctx->emitXeHardwareMask()) {
        stmts->EmitCode(ctx);
    } else {
#ifdef ISPC_XE_ENABLED
        // For Xe we insert special intrinsics at the beginning and end of unmasked region.
        // Correct execution mask will be set in CMSIMDCFLowering
        llvm::Value *oldInternalMask = ctx->XeStartUnmaskedRegion();
        stmts->EmitCode(ctx);
        ctx->XeEndUnmaskedRegion(oldInternalMask);
#endif
    }
    // Do not restore old mask if our basic block is over. This happends if we emit code
    // for something like 'unmasked{return;}', for example.
    if (ctx->GetCurrentBasicBlock() == NULL)
        return;

    ctx->SetInternalMask(oldInternalMask);
    ctx->SetFunctionMask(oldFunctionMask);
}

void UnmaskedStmt::Print(int indent) const {
    printf("%*cUnmasked Stmt", indent, ' ');
    pos.Print();
    printf("\n");

    printf("%*cStmts:\n", indent + 4, ' ');
    if (stmts != NULL)
        stmts->Print(indent + 8);
    else
        printf("NULL");
    printf("\n");
}

Stmt *UnmaskedStmt::TypeCheck() { return this; }

int UnmaskedStmt::EstimateCost() const { return COST_ASSIGN; }

///////////////////////////////////////////////////////////////////////////
// ReturnStmt

ReturnStmt::ReturnStmt(Expr *e, SourcePos p) : Stmt(p, ReturnStmtID), expr(e) {}

void ReturnStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock())
        return;

    if (ctx->InForeachLoop()) {
        Error(pos, "\"return\" statement is illegal inside a \"foreach\" loop.");
        return;
    }

    // Make sure we're not trying to return a reference to something where
    // that doesn't make sense
    const Function *func = ctx->GetFunction();
    const Type *returnType = func->GetReturnType();
    if (IsReferenceType(returnType) == true && IsReferenceType(expr->GetType()) == false) {
        const Type *lvType = expr->GetLValueType();
        if (lvType == NULL) {
            Error(expr->pos,
                  "Illegal to return non-lvalue from function "
                  "returning reference type \"%s\".",
                  returnType->GetString().c_str());
            return;
        } else if (lvType->IsUniformType() == false) {
            Error(expr->pos,
                  "Illegal to return varying lvalue type from "
                  "function returning a reference type \"%s\".",
                  returnType->GetString().c_str());
            return;
        }
    }

    ctx->SetDebugPos(pos);
    ctx->CurrentLanesReturned(expr, true);
}

Stmt *ReturnStmt::TypeCheck() { return this; }

int ReturnStmt::EstimateCost() const { return COST_RETURN; }

void ReturnStmt::Print(int indent) const {
    printf("%*cReturn Stmt", indent, ' ');
    pos.Print();
    if (expr)
        expr->Print();
    else
        printf("(void)");
    printf("\n");
}

///////////////////////////////////////////////////////////////////////////
// GotoStmt

GotoStmt::GotoStmt(const char *l, SourcePos gotoPos, SourcePos ip) : Stmt(gotoPos, GotoStmtID) {
    label = l;
    identifierPos = ip;
}

void GotoStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock())
        return;

#ifdef ISPC_XE_ENABLED
    if ((ctx->emitXeHardwareMask() && ctx->inXeSimdCF()) || ctx->VaryingCFDepth() > 0) {
#else
    if (ctx->VaryingCFDepth() > 0) {
#endif
        Error(pos, "\"goto\" statements are only legal under \"uniform\" "
                   "control flow.");
        return;
    }

    if (ctx->InForeachLoop()) {
        Error(pos, "\"goto\" statements are currently illegal inside "
                   "\"foreach\" loops.");
        return;
    }

    llvm::BasicBlock *bb = ctx->GetLabeledBasicBlock(label);
    if (bb == NULL) {
        /* Label wasn't found. Look for suggestions that are close */
        std::vector<std::string> labels = ctx->GetLabels();
        std::vector<std::string> matches = MatchStrings(label, labels);
        std::string match_output;
        if (!matches.empty()) {
            /* Print up to 5 matches. Don't want to spew too much */
            match_output += "\nDid you mean:";
            for (unsigned int i = 0; i < matches.size() && i < 5; i++)
                match_output += "\n " + matches[i] + "?";
        }

        /* Label wasn't found. Emit an error */
        Error(identifierPos, "No label named \"%s\" found in current function.%s", label.c_str(), match_output.c_str());

        return;
    }

    ctx->BranchInst(bb);
    ctx->SetCurrentBasicBlock(NULL);
}

void GotoStmt::Print(int indent) const { printf("%*cGoto label \"%s\"\n", indent, ' ', label.c_str()); }

Stmt *GotoStmt::Optimize() { return this; }

Stmt *GotoStmt::TypeCheck() { return this; }

int GotoStmt::EstimateCost() const { return COST_GOTO; }

///////////////////////////////////////////////////////////////////////////
// LabeledStmt

LabeledStmt::LabeledStmt(const char *n, Stmt *s, SourcePos p) : Stmt(p, LabeledStmtID) {
    name = n;
    stmt = s;
}

void LabeledStmt::EmitCode(FunctionEmitContext *ctx) const {
    llvm::BasicBlock *bblock = ctx->GetLabeledBasicBlock(name);
    AssertPos(pos, bblock != NULL);

    // End the current basic block with a jump to our basic block and then
    // set things up for emission to continue there.  Note that the current
    // basic block may validly be NULL going into this statement due to an
    // earlier goto that NULLed it out; that doesn't stop us from
    // re-establishing a current basic block starting at the label..
    if (ctx->GetCurrentBasicBlock() != NULL)
        ctx->BranchInst(bblock);
    ctx->SetCurrentBasicBlock(bblock);

    if (stmt != NULL)
        stmt->EmitCode(ctx);
}

void LabeledStmt::Print(int indent) const {
    printf("%*cLabel \"%s\"\n", indent, ' ', name.c_str());
    if (stmt != NULL)
        stmt->Print(indent);
}

Stmt *LabeledStmt::Optimize() { return this; }

Stmt *LabeledStmt::TypeCheck() {
    if (!isalpha(name[0]) || name[0] == '_') {
        Error(pos, "Label must start with either alphabetic character or '_'.");
        return NULL;
    }
    for (unsigned int i = 1; i < name.size(); ++i) {
        if (!isalnum(name[i]) && name[i] != '_') {
            Error(pos, "Character \"%c\" is illegal in labels.", name[i]);
            return NULL;
        }
    }
    return this;
}

int LabeledStmt::EstimateCost() const { return 0; }

///////////////////////////////////////////////////////////////////////////
// StmtList

void StmtList::EmitCode(FunctionEmitContext *ctx) const {
    ctx->StartScope();
    ctx->SetDebugPos(pos);
    for (unsigned int i = 0; i < stmts.size(); ++i)
        if (stmts[i])
            stmts[i]->EmitCode(ctx);
    ctx->EndScope();
}

Stmt *StmtList::TypeCheck() { return this; }

int StmtList::EstimateCost() const { return 0; }

void StmtList::Print(int indent) const {
    printf("%*cStmt List", indent, ' ');
    pos.Print();
    printf(":\n");
    for (unsigned int i = 0; i < stmts.size(); ++i)
        if (stmts[i])
            stmts[i]->Print(indent + 4);
}

///////////////////////////////////////////////////////////////////////////
// PrintStmt

PrintStmt::PrintStmt(const std::string &f, Expr *v, SourcePos p) : Stmt(p, PrintStmtID), format(f), values(v) {}

/* Because the pointers to values that are passed to __do_print() are all
   void *s (and because ispc print() formatting strings statements don't
   encode types), we pass along a string to __do_print() where the i'th
   character encodes the type of the i'th value to be printed. The encoding
   is defined by getEncoding4Uniform<T> and getEncding4Varying<T> functions.
 */
static char lEncodeType(const Type *t) {
    if (Type::Equal(t, AtomicType::UniformBool))
        return PrintInfo::getEncoding4Uniform<bool>();
    if (Type::Equal(t, AtomicType::VaryingBool))
        return PrintInfo::getEncoding4Varying<bool>();
    if (Type::Equal(t, AtomicType::UniformInt32))
        return PrintInfo::getEncoding4Uniform<int>();
    if (Type::Equal(t, AtomicType::VaryingInt32))
        return PrintInfo::getEncoding4Varying<int>();
    if (Type::Equal(t, AtomicType::UniformUInt32))
        return PrintInfo::getEncoding4Uniform<unsigned>();
    if (Type::Equal(t, AtomicType::VaryingUInt32))
        return PrintInfo::getEncoding4Varying<unsigned>();
    if (Type::Equal(t, AtomicType::UniformFloat))
        return PrintInfo::getEncoding4Uniform<float>();
    if (Type::Equal(t, AtomicType::VaryingFloat))
        return PrintInfo::getEncoding4Varying<float>();
    if (Type::Equal(t, AtomicType::UniformInt64))
        return PrintInfo::getEncoding4Uniform<long long>();
    if (Type::Equal(t, AtomicType::VaryingInt64))
        return PrintInfo::getEncoding4Varying<long long>();
    if (Type::Equal(t, AtomicType::UniformUInt64))
        return PrintInfo::getEncoding4Uniform<unsigned long long>();
    if (Type::Equal(t, AtomicType::VaryingUInt64))
        return PrintInfo::getEncoding4Varying<unsigned long long>();
    if (Type::Equal(t, AtomicType::UniformDouble))
        return PrintInfo::getEncoding4Uniform<double>();
    if (Type::Equal(t, AtomicType::VaryingDouble))
        return PrintInfo::getEncoding4Varying<double>();
    if (CastType<PointerType>(t) != NULL) {
        if (t->IsUniformType())
            return PrintInfo::getEncoding4Uniform<void *>();
        else
            return PrintInfo::getEncoding4Varying<void *>();
    } else
        return '\0';
}

struct ExprWithType {
    Expr *expr;
    char type;
};

static ExprWithType lProcessPrintArgType(Expr *expr) {
    const Type *type = expr->GetType();
    if (type == NULL)
        return {NULL, '\0'};

    if (CastType<ReferenceType>(type) != NULL) {
        expr = new RefDerefExpr(expr, expr->pos);
        type = expr->GetType();
        if (type == NULL)
            return {NULL, '\0'};
    }

    // Just int8 and int16 types to int32s...
    // Also ensure 'varying bool' is excluded since it's baseType can be one
    // of these types.
    const Type *baseType = type->GetAsNonConstType()->GetAsUniformType();
    if ((Type::Equal(baseType, AtomicType::UniformInt8) || Type::Equal(baseType, AtomicType::UniformUInt8) ||
         Type::Equal(baseType, AtomicType::UniformInt16) || Type::Equal(baseType, AtomicType::UniformUInt16)) &&
        !type->IsBoolType()) {
        expr = new TypeCastExpr(type->IsUniformType() ? AtomicType::UniformInt32 : AtomicType::VaryingInt32, expr,
                                expr->pos);
        type = expr->GetType();
    }

    if (Type::Equal(baseType, AtomicType::UniformFloat16)) {
        expr = new TypeCastExpr(type->IsUniformType() ? AtomicType::UniformFloat : AtomicType::VaryingFloat, expr,
                                expr->pos);
        type = expr->GetType();
    }

    char t = lEncodeType(type->GetAsNonConstType());
    if (t == '\0') {
        Error(expr->pos,
              "Only atomic types are allowed in print statements; "
              "type \"%s\" is illegal.",
              type->GetString().c_str());
        return {NULL, '\0'};
    }
    if (type->IsBoolType()) {
        // Blast bools to ints, but do it here to preserve encoding for
        // printing 'true' or 'false'
        expr = new TypeCastExpr(type->IsUniformType() ? AtomicType::UniformInt32 : AtomicType::VaryingInt32, expr,
                                expr->pos);
    }
    return {expr, t};
}

// Returns pointer to __do_print function
static llvm::Function *getPrintImplFunc() {
    Assert(g->target->isXeTarget() == false);
    llvm::Function *printImplFunc = m->module->getFunction("__do_print");
    return printImplFunc;
}

// Check if number of requested arguments in format string corresponds to actual number of arguments
static bool checkFormatString(const std::string &format, const int nArgs, const SourcePos &pos) {
    // We do not allow escape percent sign in ISPC as %%, so treat it as two args
    const int argsInFormat = std::count(format.begin(), format.end(), '%');
    if (nArgs < argsInFormat) {
        Error(pos, "Not enough arguments are provided in print call");
        return false;
    } else if (nArgs > argsInFormat) {
        Error(pos, "Too much arguments are provided in print call");
        return false;
    }
    return true;
}
#ifdef ISPC_XE_ENABLED
// Builds args for OCL printf function based on ISPC print args.
class PrintArgsBuilder {
    // properly dereferenced and size extended value expressions
    std::vector<ExprWithType> argExprs;
    FunctionEmitContext *ctx;

    struct AdditionalData {
        llvm::Value *mask;
        enum { LeftParenthesisIdx = 0, RightParenthesisIdx, EmptyIdx, FalseIdx, TrueIdx, NumStrings };
        std::array<llvm::Value *, NumStrings> strings;

        AdditionalData() { mask = NULL; }
        AdditionalData(FunctionEmitContext *ctx) {
            if (ctx->emitXeHardwareMask())
                mask = ctx->XeSimdCFPredicate(LLVMMaskAllOn);
            else
                mask = ctx->GetFullMask();
            strings[AdditionalData::LeftParenthesisIdx] =
                ctx->XeGetOrCreateConstantString("((", "ispc.print.left.parenthesis");
            strings[AdditionalData::RightParenthesisIdx] =
                ctx->XeGetOrCreateConstantString("))", "ispc.print.right.parenthesis");
            strings[AdditionalData::EmptyIdx] = ctx->XeGetOrCreateConstantString("", "ispc.print.empty");
            strings[AdditionalData::FalseIdx] = ctx->XeGetOrCreateConstantString("false", "ispc.print.false");
            strings[AdditionalData::TrueIdx] = ctx->XeGetOrCreateConstantString("true", "ispc.print.true");
        }
    };
    AdditionalData data;

  public:
    PrintArgsBuilder() : ctx{NULL} {}
    PrintArgsBuilder(FunctionEmitContext *ctxIn) : argExprs{}, ctx{ctxIn}, data{ctxIn} {}

    template <typename Iter>
    PrintArgsBuilder(Iter first, Iter last, FunctionEmitContext *ctxIn) : PrintArgsBuilder{ctxIn} {
        std::transform(first, last, std::back_inserter(argExprs),
                       [](Expr *expr) { return lProcessPrintArgType(expr); });
        std::for_each(argExprs.cbegin(), argExprs.cend(),
                      [](const ExprWithType &elem) { Assert(elem.expr && "must have all values processed"); });
    }

    // Returns new args builder with subset of original args.
    // Subset is defined with pair of indexes, element with \p end index is not included.
    PrintArgsBuilder extract(int beg, int end) const {
        Assert(beg >= 0 && beg <= argExprs.size() && end >= 0 && end <= argExprs.size() &&
               "wrong argument: index is out of bound");
        Assert(beg <= end && "wrong arguments: beg must preceed end");
        PrintArgsBuilder extraction(ctx, data);
        std::copy(std::next(argExprs.begin(), beg), std::next(argExprs.begin(), end),
                  std::back_inserter(extraction.argExprs));
        return extraction;
    }

    // Combine all arg types into a continuous string.
    std::string generateArgTypes() const {
        std::string argTypes;
        std::transform(argExprs.cbegin(), argExprs.cend(), std::back_inserter(argTypes),
                       [](const ExprWithType &argInfo) { return argInfo.type; });
        return argTypes;
    }

    // Emit code for OCL printf arguments.
    // Each generated arg is returned in the vector.
    std::vector<llvm::Value *> emitArgCode() const {
        if (argExprs.empty())
            return {};

        std::vector<llvm::Value *> Args;
        // It would require at least the same amount of args. More if there're vector args.
        Args.reserve(argExprs.size());
        for (const ExprWithType &argInfo : argExprs)
            writeRawArg(*argInfo.expr->GetValue(ctx), static_cast<PrintInfo::Encoding>(argInfo.type),
                        std::back_inserter(Args));
        return Args;
    }

  private:
    PrintArgsBuilder(FunctionEmitContext *ctxIn, AdditionalData data) : argExprs{}, ctx{ctxIn}, data{std::move(data)} {}

    // Emit code for ISPC print uniform arg.
    // Most arg types are unchanged. Boolean arg should be transformed into string argument.
    // Pointers to generated llvm::Value are stored into output iterator \p OutIt.
    template <typename OutIter>
    OutIter writeRawUniformArg(llvm::Value &rawArg, PrintInfo::Encoding type, OutIter OutIt) const {
        if (type != PrintInfo::getEncoding4Uniform<bool>()) {
            *OutIt++ = &rawArg;
            return OutIt;
        }
        auto *argAsPred = ctx->CmpInst(llvm::Instruction::OtherOps::ICmp, llvm::CmpInst::Predicate::ICMP_NE, &rawArg,
                                       LLVMInt32(0), "print.arg.bool.cast");
        auto *argAsStr = ctx->SelectInst(argAsPred, data.strings[AdditionalData::TrueIdx],
                                         data.strings[AdditionalData::FalseIdx], "print.arg.bool.str");
        *OutIt++ = argAsStr;
        return OutIt;
    }

    // Emit code for ISPC print varying arg.
    // Each element of a vector is emitted as a separate OCL printf argument. Plus it is surounded by two string
    // arguments to print additional parantheses when the corresponding lane is off. Pointers to generated llvm::Value
    // are stored into output iterator \p OutIt.
    template <typename OutIter>
    OutIter writeRawVaryingArg(llvm::Value &rawArg, PrintInfo::Encoding type, OutIter OutIt) const {
        auto width = g->target->getVectorWidth();
        for (int idx = 0; idx != width; ++idx) {
            auto *isLaneOn = ctx->ExtractInst(data.mask, idx, "print.arg.lane");
            auto *leftParenthesis =
                ctx->SelectInst(isLaneOn, data.strings[AdditionalData::EmptyIdx],
                                data.strings[AdditionalData::LeftParenthesisIdx], "print.arg.left.par");
            auto *rightParenthesis =
                ctx->SelectInst(isLaneOn, data.strings[AdditionalData::EmptyIdx],
                                data.strings[AdditionalData::RightParenthesisIdx], "print.arg.right.par");
            auto *argElement = ctx->ExtractInst(&rawArg, idx, "print.arg.elem");
            *OutIt++ = leftParenthesis;
            OutIt = writeRawUniformArg(*argElement, PrintInfo::getCorrespondingEncoding4Uniform(type), OutIt);
            *OutIt++ = rightParenthesis;
        }
        return OutIt;
    }

    // Emit printf OCL args (one or more) based on the single ISPC print arg \p rawArg and its \p type.
    // Pointers to generated llvm::Value are stored into output iterator \p OutIt.
    template <typename OutIter>
    OutIter writeRawArg(llvm::Value &rawArg, PrintInfo::Encoding type, OutIter OutIt) const {
        if (PrintInfo::isUniformEncoding(type))
            return writeRawUniformArg(rawArg, type, OutIt);
        return writeRawVaryingArg(rawArg, type, OutIt);
    }
};

// When one print is split in several smaller ones,
// this structure will hold info about a split.
struct PrintSliceInfo {
    // format holds ISPC-style format string (with just '%')
    std::string format_;
    PrintArgsBuilder args_;
};

class PrintLZFormatStrBuilder {
    std::array<std::string, PrintInfo::Encoding::Size> specifiers;
    const int width;

  public:
    PrintLZFormatStrBuilder(int widthIn) : width{widthIn} {}

    // Based on original ISPC format string, and encoded arg types
    // generates printf format string.
    std::string get(const std::string &ISPCFormat, const std::string &argTypes, const SourcePos &pos) {
        std::string format;
        if (!checkFormatString(ISPCFormat, argTypes.size(), pos))
            return "";
        format.reserve(ISPCFormat.size());
        auto curISPCFormatIt = ISPCFormat.begin();
        for (auto type : argTypes) {
            auto percentIt = std::find(curISPCFormatIt, ISPCFormat.end(), '%');
            if (percentIt == ISPCFormat.end())
                Error(pos, "Too much arguments are provided in print call");
            format.append(curISPCFormatIt, percentIt);
            format.append(getOrCreateSpecifier(static_cast<PrintInfo::Encoding>(type)));
            curISPCFormatIt = std::next(percentIt);
        }
        if (std::any_of(curISPCFormatIt, ISPCFormat.end(), [](char ch) { return ch == '%'; }))
            Error(pos, "Not enough arguments are provided in print call");
        format.append(curISPCFormatIt, ISPCFormat.end());
        return format;
    }

    const std::string &getOrCreateSpecifier(PrintInfo::Encoding type) {
        assertEncoding(type);
        auto &specifier = getSpecifier(type);
        if (specifier.empty())
            return createSpecifier(type);
        return specifier;
    }

  private:
    static void assertEncoding(PrintInfo::Encoding type) {
        Assert(type >= PrintInfo::Encoding::Bool && type <= PrintInfo::Encoding::VecPtr &&
               "wrong argument: unsupported type");
    }

    static void assertEncoding4Uniform(PrintInfo::Encoding type) {
        Assert(type >= PrintInfo::Encoding::Bool && type < PrintInfo::Encoding::VecBool &&
               "wrong argument: unsupported type");
    }

    static void assertEncoding4Varying(PrintInfo::Encoding type) {
        Assert(type >= PrintInfo::Encoding::VecBool && type <= PrintInfo::Encoding::VecPtr &&
               "wrong argument: unsupported type");
    }

    // helper functor to generate specifier for uniform type
    struct FillSpecifier4Uniform {
        std::string &str;
        FillSpecifier4Uniform(std::string &strIn) : str(strIn) {}
        template <typename T> void call() { str = PrintInfo::type2Specifier<T>(); }
    };

    // tip: don't access specifiers field directly, use this function.
    std::string &accessSpecifier(PrintInfo::Encoding type) {
        assertEncoding(type);
        return specifiers[type - PrintInfo::Bool];
    }

    const std::string &getSpecifier(PrintInfo::Encoding type) const {
        assertEncoding(type);
        return const_cast<PrintLZFormatStrBuilder *>(this)->accessSpecifier(type);
    }

    const std::string &createSpecifier4Uniform(PrintInfo::Encoding type) {
        assertEncoding4Uniform(type);
        switchEncoding4Uniform(type, FillSpecifier4Uniform{accessSpecifier(type)});
        return getSpecifier(type);
    }

    const std::string &getOrCreateSpecifier4Uniform(PrintInfo::Encoding type) {
        assertEncoding4Uniform(type);
        auto &specifier = getSpecifier(type);
        if (specifier.empty())
            return createSpecifier4Uniform(type);
        return specifier;
    }

    const std::string &createSpecifier4Varying(PrintInfo::Encoding type) {
        assertEncoding4Varying(type);
        const std::string &uniform = getOrCreateSpecifier4Uniform(getCorrespondingEncoding4Uniform(type));
        std::stringstream ss;
        ss << "[";
        for (int i = 0; i < width - 1; ++i)
            ss << "%s" << uniform << "%s,";
        ss << "%s" << uniform << "%s]";
        accessSpecifier(type) = ss.str();
        return getSpecifier(type);
    }

    const std::string &createSpecifier(PrintInfo::Encoding type) {
        assertEncoding(type);
        if (type < PrintInfo::Encoding::VecBool)
            return createSpecifier4Uniform(type);
        return createSpecifier4Varying(type);
    }
};

// Finds a prefix of a format string with a valid weight.
//
// Arguments:
//    [\p formatFirst, \p formatLast) format string defined with a range
//    [\p typeWeightFirst, ...) range of weights of every type to be printed,
//      only begin of the range is required, length of the range must correspond to
//      the number of '%' in format string. Weight here is meant to represent
//      the length of the string, with which '%' will be replaced. In other
//      words weight of every char except '%' is 1, and the weight of every
//      '%' char is taken from the range.
//    \p LZPrintFormatLimit - limit on the weight of the resulting string.
//
// Iterator to the provided format string such that string [\p formatFirst, returned iter)
// meets the limit is returned.
// Iterator to the element past the last weight element used is returned. When no weight
// info is used unchanged \p typeWeightFirst is returned.
template <typename FormatIt, typename TypeWeightIt>
std::tuple<FormatIt, TypeWeightIt> splitValidFormat(FormatIt formatFirst, FormatIt formatLast,
                                                    TypeWeightIt typeWeightFirst, int LZPrintFormatLimit,
                                                    const std::vector<int> argWeights) {
    int sum = 0;
    // space for '\0'
    --LZPrintFormatLimit;
    for (; formatFirst != formatLast; ++formatFirst) {
        char curCh = *formatFirst;
        // Check that typeWeightFirst can be safely derefrenced here
        if ((curCh == '%') && (typeWeightFirst != argWeights.end())) {
            sum += *typeWeightFirst;
            if (sum <= LZPrintFormatLimit)
                ++typeWeightFirst;
        } else
            ++sum;
        if (sum > LZPrintFormatLimit)
            return {formatFirst, typeWeightFirst};
    }
    return {formatFirst, typeWeightFirst};
}

// Splits original print into several prints with valid length format strings.
static std::vector<PrintSliceInfo> getPrintSlices(const std::string &format, PrintLZFormatStrBuilder &formatBuilder,
                                                  const PrintArgsBuilder &args, const int LZPrintFormatLimit,
                                                  const SourcePos &pos) {
    auto argTypes = args.generateArgTypes();
    std::vector<PrintSliceInfo> printSlices;
    if (!checkFormatString(format, argTypes.size(), pos)) {
        return printSlices;
    }
    std::vector<int> argWeights(argTypes.size());
    std::transform(argTypes.begin(), argTypes.end(), argWeights.begin(), [&formatBuilder](char type) {
        return formatBuilder.getOrCreateSpecifier(static_cast<PrintInfo::Encoding>(type)).size();
    });
    auto firstArgWeight = argWeights.begin();
    auto curArgWeight = firstArgWeight;
    for (auto curFormat = format.begin(), lastFormat = format.end(); curFormat != lastFormat;) {
        auto prevFormat = curFormat;
        auto prevArgWeight = curArgWeight;
        std::tie(curFormat, curArgWeight) =
            splitValidFormat(curFormat, lastFormat, curArgWeight, LZPrintFormatLimit, argWeights);
        Assert(curFormat > prevFormat && "haven't managed to split format string");
        printSlices.push_back({std::string(prevFormat, curFormat),
                               args.extract(prevArgWeight - firstArgWeight, curArgWeight - firstArgWeight)});
    }
    return printSlices;
}

// prepares arguments for __spirv_ocl_printf function
static std::vector<llvm::Value *> getOCLPrintfArgs(const std::string &format, PrintLZFormatStrBuilder &formatBuilder,
                                                   const PrintArgsBuilder &args, FunctionEmitContext *ctx,
                                                   const SourcePos &pos) {
    std::vector<llvm::Value *> allArgs;
    auto argTypes = args.generateArgTypes();
    allArgs.push_back(ctx->XeCreateConstantString(formatBuilder.get(format, argTypes, pos), "lz_format_str"));
    auto valueArgs = args.emitArgCode();
    std::move(valueArgs.begin(), valueArgs.end(), std::back_inserter(allArgs));
    return allArgs;
}

static PrintArgsBuilder getPrintArgsBuilder(Expr *values, FunctionEmitContext *ctx) {
    if (values == NULL)
        return PrintArgsBuilder(ctx);
    else {
        ExprList *elist = llvm::dyn_cast<ExprList>(values);
        if (elist)
            return PrintArgsBuilder{elist->exprs.begin(), elist->exprs.end(), ctx};
        else
            return PrintArgsBuilder{&values, &values + 1, ctx};
    }
}

// This name should be also properly mangled. It happens later.
static llvm::FunctionCallee getSPIRVOCLPrintfDecl() {
    auto *PrintfTy = llvm::FunctionType::get(LLVMTypes::Int32Type,
                                             llvm::PointerType::get(LLVMTypes::Int8Type, /* const addrspace */ 2),
                                             /* isVarArg */ true);
    return m->module->getOrInsertFunction("__spirv_ocl_printf", PrintfTy);
}

static void emitCode4LZPrintSlice(const PrintSliceInfo &printSlice, PrintLZFormatStrBuilder &formatBuilder,
                                  FunctionEmitContext *ctx, const SourcePos &pos) {
    auto printImplArgs = getOCLPrintfArgs(printSlice.format_, formatBuilder, printSlice.args_, ctx, pos);
    auto printImplFunc = getSPIRVOCLPrintfDecl();
    Assert(printImplFunc && "__spirv_ocl_printf declaration wasn't created");
    ctx->CallInst(printImplFunc.getCallee(), NULL, printImplArgs, "");
}

void PrintStmt::emitCode4LZ(FunctionEmitContext *ctx) const {
    auto allArgs = getPrintArgsBuilder(values, ctx);
    PrintLZFormatStrBuilder formatBuilder(g->target->getVectorWidth());
    auto printSlices = getPrintSlices(format, formatBuilder, allArgs, PrintInfo::LZMaxFormatStrSize, pos);
    for (const auto &printSlice : printSlices)
        emitCode4LZPrintSlice(printSlice, formatBuilder, ctx, pos);
}

#endif // ISPC_XE_ENABLED

/** Given an Expr for a value to be printed, emit the code to evaluate the
    expression and store the result to alloca's memory.  Update the
    argTypes string with the type encoding for this expression.
 */
static llvm::Value *lEmitPrintArgCode(Expr *expr, FunctionEmitContext *ctx) {
    const Type *type = expr->GetType();

    llvm::Type *llvmExprType = type->LLVMType(g->ctx);
    llvm::Value *ptr = ctx->AllocaInst(llvmExprType, "print_arg");
    llvm::Value *val = expr->GetValue(ctx);
    if (!val)
        return NULL;
    ctx->StoreInst(val, ptr);

    ptr = ctx->BitCastInst(ptr, LLVMTypes::VoidPointerType);
    return ptr;
}

static bool lProcessPrintArg(Expr *expr, FunctionEmitContext *ctx, llvm::Value *argPtrArray, int offset,
                             std::string &argTypes) {
    if (!expr)
        return false;
    auto exprType = lProcessPrintArgType(expr);
    expr = exprType.expr;
    char type = exprType.type;
    if (!expr)
        return false;
    argTypes.push_back(type);
    llvm::Value *ptr = lEmitPrintArgCode(expr, ctx);
    if (!ptr)
        return false;
    llvm::Value *arrayPtr = ctx->AddElementOffset(argPtrArray, offset, NULL);
    ctx->StoreInst(ptr, arrayPtr);
    return true;
}

// prepares arguments for __do_print function
std::vector<llvm::Value *> PrintStmt::getDoPrintArgs(FunctionEmitContext *ctx) const {
    std::vector<llvm::Value *> doPrintArgs(STD_NUM_IDX);
    std::string argTypes;

    if (values == NULL) {
        // Check requested format
        checkFormatString(format, 0, pos);
        llvm::Type *ptrPtrType = llvm::PointerType::get(LLVMTypes::VoidPointerType, 0);
        doPrintArgs[ARGS_IDX] = llvm::Constant::getNullValue(ptrPtrType);
    } else {
        // Get the values passed to the print() statement evaluated and
        // stored in memory so that we set up the array of pointers to them
        // for the 5th __do_print() argument
        ExprList *elist = llvm::dyn_cast<ExprList>(values);
        int nArgs = elist ? elist->exprs.size() : 1;
        // Check requested format
        checkFormatString(format, nArgs, pos);
        // Allocate space for the array of pointers to values to be printed
        llvm::Type *argPtrArrayType = llvm::ArrayType::get(LLVMTypes::VoidPointerType, nArgs);
        llvm::Value *argPtrArray = ctx->AllocaInst(argPtrArrayType, "print_arg_ptrs");
        // Store the array pointer as a void **, which is what __do_print()
        // expects
        doPrintArgs[ARGS_IDX] = ctx->BitCastInst(argPtrArray, llvm::PointerType::get(LLVMTypes::VoidPointerType, 0));

        // Now, for each of the arguments, emit code to evaluate its value
        // and store the value into alloca's storage.  Then store the
        // pointer to the alloca's storage into argPtrArray.
        if (elist) {
            for (unsigned int i = 0; i < elist->exprs.size(); ++i) {
                Expr *expr = elist->exprs[i];
                if (!lProcessPrintArg(expr, ctx, argPtrArray, i, argTypes)) {
                    return {};
                }
            }
        } else {
            if (lProcessPrintArg(values, ctx, argPtrArray, 0, argTypes)) {
                return {};
            }
        }
    }
    llvm::Value *mask = ctx->GetFullMask();
    // Set up the rest of the parameters to it
    doPrintArgs[FORMAT_IDX] = ctx->GetStringPtr(format);
    doPrintArgs[TYPES_IDX] = ctx->GetStringPtr(argTypes);
    doPrintArgs[WIDTH_IDX] = LLVMInt32(g->target->getVectorWidth());
    doPrintArgs[MASK_IDX] = ctx->LaneMask(mask);
    return doPrintArgs;
}

/* PrintStmt works closely with the __do_print() function implemented in
   the builtins-c-cpu.cpp file. In particular, the EmitCode() method here needs to
   take the arguments passed to it from ispc and generate a valid call to
   __do_print() with the information that __do_print() then needs to do the
   actual printing work at runtime.
 */
void PrintStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock())
        return;
    ctx->SetDebugPos(pos);
#ifdef ISPC_XE_ENABLED
    if (g->target->isXeTarget()) {
        emitCode4LZ(ctx);
        return;
    }
#endif /* ISPC_XE_ENABLED */
    auto printImplArgs = getDoPrintArgs(ctx);
    Assert(!printImplArgs.empty() && "Haven't managed to produce __do_print args");
    auto printImplFunc = getPrintImplFunc();
    AssertPos(pos, printImplFunc);
    ctx->CallInst(printImplFunc, NULL, printImplArgs, "");
}

void PrintStmt::Print(int indent) const { printf("%*cPrint Stmt (%s)", indent, ' ', format.c_str()); }

Stmt *PrintStmt::TypeCheck() { return this; }

int PrintStmt::EstimateCost() const { return COST_FUNCALL; }

///////////////////////////////////////////////////////////////////////////
// AssertStmt

AssertStmt::AssertStmt(const std::string &msg, Expr *e, SourcePos p) : Stmt(p, AssertStmtID), message(msg), expr(e) {}

void AssertStmt::EmitAssertCode(FunctionEmitContext *ctx, const Type *type) const {
    bool isUniform = type->IsUniformType();

    // The actual functionality to do the check and then handle failure is
    // done via a builtin written in bitcode in builtins/util.m4.
    llvm::Function *assertFunc =
        isUniform ? m->module->getFunction("__do_assert_uniform") : m->module->getFunction("__do_assert_varying");
    AssertPos(pos, assertFunc != NULL);

    char *errorString;
    if (asprintf(&errorString, "%s:%d:%d: Assertion failed: %s \n", pos.name, pos.first_line, pos.first_column,
                 message.c_str()) == -1) {
        Error(pos, "Fatal error when generating assert string: asprintf() "
                   "unable to allocate memory!");
        return;
    }

    std::vector<llvm::Value *> args;
#ifdef ISPC_XE_ENABLED
    if (g->target->isXeTarget()) {
        PrintLZFormatStrBuilder formatBuilder(g->target->getVectorWidth());
        args.push_back(ctx->XeCreateConstantString(errorString, "lz_format_str"));
    } else
#endif
        args.push_back(ctx->GetStringPtr(errorString));
    llvm::Value *exprValue = expr->GetValue(ctx);
    if (exprValue == NULL) {
        free(errorString);
        AssertPos(pos, m->errorCount > 0);
        return;
    }
    args.push_back(exprValue);
#ifdef ISPC_XE_ENABLED
    if (ctx->emitXeHardwareMask())
        // This will create mask according to current EM on SIMD CF Lowering.
        // The result will be like       mask = select (EM, AllOn, AllFalse)
        args.push_back(ctx->XeSimdCFPredicate(LLVMMaskAllOn));
    else
#endif
        args.push_back(ctx->GetFullMask());
    ctx->CallInst(assertFunc, NULL, args, "");

    free(errorString);
}

void AssertStmt::EmitAssumeCode(FunctionEmitContext *ctx, const Type *type) const {
    bool isUniform = type->IsUniformType();

    // Currently, we insert an assume only for uniform conditions.
    if (!isUniform) {
        return;
    }

    // The actual functionality to insert an 'llvm.assume' intrinsic is
    // done via a builtin written in bitcode in builtins/util.m4.
    llvm::Function *assumeFunc = m->module->getFunction("__do_assume_uniform");
    AssertPos(pos, assumeFunc != NULL);

    llvm::Value *exprValue = expr->GetValue(ctx);
    if (exprValue == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return;
    }
    ctx->CallInst(assumeFunc, NULL, exprValue, "");
}

void AssertStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock())
        return;

    const Type *type;
    if (expr == NULL || (type = expr->GetType()) == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return;
    }
    if (g->opt.disableAsserts) {
        EmitAssumeCode(ctx, type);
    } else {
        EmitAssertCode(ctx, type);
    }
}

void AssertStmt::Print(int indent) const { printf("%*cAssert Stmt (%s)", indent, ' ', message.c_str()); }

Stmt *AssertStmt::TypeCheck() {
    const Type *type;
    if (expr && (type = expr->GetType()) != NULL) {
        bool isUniform = type->IsUniformType();
        expr = TypeConvertExpr(expr, isUniform ? AtomicType::UniformBool : AtomicType::VaryingBool,
                               "\"assert\" statement");
        if (expr == NULL)
            return NULL;
    }
    return this;
}

int AssertStmt::EstimateCost() const { return COST_ASSERT; }

///////////////////////////////////////////////////////////////////////////
// DeleteStmt

DeleteStmt::DeleteStmt(Expr *e, SourcePos p) : Stmt(p, DeleteStmtID) { expr = e; }

void DeleteStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (g->target->isXeTarget()) {
        Error(pos, "\"delete\" statement is not supported for Xe targets yet.");
        return;
    }

    if (!ctx->GetCurrentBasicBlock())
        return;

    const Type *exprType;
    if (expr == NULL || ((exprType = expr->GetType()) == NULL)) {
        AssertPos(pos, m->errorCount > 0);
        return;
    }

    llvm::Value *exprValue = expr->GetValue(ctx);
    if (exprValue == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return;
    }

    // Typechecking should catch this
    AssertPos(pos, CastType<PointerType>(exprType) != NULL);

    if (exprType->IsUniformType()) {
        // For deletion of a uniform pointer, we just need to cast the
        // pointer type to a void pointer type, to match what
        // __delete_uniform() from the builtins expects.
        exprValue = ctx->BitCastInst(exprValue, LLVMTypes::VoidPointerType, "ptr_to_void");
        llvm::Function *func;
        if (g->target->is32Bit()) {
            func = m->module->getFunction("__delete_uniform_32rt");
        } else {
            func = m->module->getFunction("__delete_uniform_64rt");
        }
        AssertPos(pos, func != NULL);

        ctx->CallInst(func, NULL, exprValue, "");
    } else {
        // Varying pointers are arrays of ints, and __delete_varying()
        // takes a vector of i64s (even for 32-bit targets).  Therefore, we
        // only need to extend to 64-bit values on 32-bit targets before
        // calling it.
        llvm::Function *func;
        if (g->target->is32Bit()) {
            func = m->module->getFunction("__delete_varying_32rt");
        } else {
            func = m->module->getFunction("__delete_varying_64rt");
        }
        AssertPos(pos, func != NULL);
        if (g->target->is32Bit())
            exprValue = ctx->ZExtInst(exprValue, LLVMTypes::Int64VectorType, "ptr_to_64");
        ctx->CallInst(func, NULL, exprValue, "");
    }
}

void DeleteStmt::Print(int indent) const { printf("%*cDelete Stmt", indent, ' '); }

Stmt *DeleteStmt::TypeCheck() {
    const Type *exprType;
    if (expr == NULL || ((exprType = expr->GetType()) == NULL))
        return NULL;

    if (CastType<PointerType>(exprType) == NULL) {
        Error(pos, "Illegal to delete non-pointer type \"%s\".", exprType->GetString().c_str());
        return NULL;
    }

    return this;
}

int DeleteStmt::EstimateCost() const { return COST_DELETE; }
