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

/** @file stmt.cpp
    @brief File with definitions classes related to statements in the language
*/

#include "stmt.h"
#include "ctx.h"
#include "util.h"
#include "expr.h"
#include "type.h"
#include "decl.h"
#include "sym.h"
#include "module.h"
#include "llvmutil.h"

#include <stdio.h>
#include <map>

#include <llvm/Module.h>
#include <llvm/Function.h>
#include <llvm/Type.h>
#include <llvm/DerivedTypes.h>
#include <llvm/LLVMContext.h>
#include <llvm/Metadata.h>
#include <llvm/Instructions.h>
#include <llvm/CallingConv.h>
#include <llvm/Support/IRBuilder.h>
#include <llvm/Support/raw_ostream.h>

///////////////////////////////////////////////////////////////////////////
// ExprStmt

ExprStmt::ExprStmt(Expr *e, SourcePos p) 
  : Stmt(p) {
    expr = e;
}

void
ExprStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock()) 
        return;

    ctx->SetDebugPos(pos);
    if (expr) 
        expr->GetValue(ctx);
}


Stmt *
ExprStmt::Optimize() {
    if (expr) 
        expr = expr->Optimize();
    return this;
}


Stmt *
ExprStmt::TypeCheck() {
    if (expr) 
        expr = expr->TypeCheck();
    return this;
}


void
ExprStmt::Print(int indent) const {
    if (!expr) 
        return;

    printf("%*c", indent, ' ');
    printf("Expr stmt: ");
    pos.Print();
    expr->Print();
    printf("\n");
}


///////////////////////////////////////////////////////////////////////////
// DeclStmt

DeclStmt::DeclStmt(SourcePos p, Declaration *d, SymbolTable *s)
        : Stmt(p), declaration(d) {
    declaration->AddSymbols(s);
}


/** Utility routine that emits code to initialize a symbol given an
    initializer expression.

    @param lvalue    Memory location of storage for the symbol's data
    @param symName   Name of symbol (used in error messages)
    @param type      Type of variable being initialized
    @param initExpr  Expression for the initializer
    @param ctx       FunctionEmitContext to use for generating instructions
    @param pos       Source file position of the variable being initialized
*/
static void
lInitSymbol(llvm::Value *lvalue, const char *symName, const Type *type, 
            Expr *initExpr, FunctionEmitContext *ctx, SourcePos pos) {
    if (initExpr == NULL) {
        // Initialize things without initializers to the undefined value.
        // To auto-initialize everything to zero, replace 'UndefValue' with
        // 'NullValue' in the below
        LLVM_TYPE_CONST llvm::Type *ltype = type->LLVMType(g->ctx);
        ctx->StoreInst(llvm::UndefValue::get(ltype), lvalue);
        return;
    }

    // If the initializer is a straight up expression that isn't an
    // ExprList, then we'll see if we can type convert it to the type of
    // the variable.
    if (dynamic_cast<ExprList *>(initExpr) == NULL) {
        Expr *tcInit = initExpr->TypeConv(type, "inititalizer", true);
        if (tcInit != NULL) {
            llvm::Value *initializerValue = tcInit->GetValue(ctx);
            if (initializerValue != NULL)
                // Bingo; store the value in the variable's storage
                ctx->StoreInst(initializerValue, lvalue);
            return;
        }
    }

    // Atomic types and enums can't be initialized with { ... } initializer
    // expressions, so print an error and return if that's what we've got
    // here..
    if (dynamic_cast<const AtomicType *>(type) != NULL ||
        dynamic_cast<const EnumType *>(type) != NULL) {
        if (dynamic_cast<ExprList *>(initExpr) != NULL)
            Error(initExpr->pos, "Expression list initializers can't be used for "
                  "variable \"%s\' with type \"%s\".", symName,
                  type->GetString().c_str());
        return;
    }

    const ReferenceType *rt = dynamic_cast<const ReferenceType *>(type);
    if (rt) {
        if (!Type::Equal(initExpr->GetType(), rt)) {
            Error(initExpr->pos, "Initializer for reference type \"%s\" must have same "
                  "reference type itself. \"%s\" is incompatible.", 
                  rt->GetString().c_str(), initExpr->GetType()->GetString().c_str());
            return;
        }

        llvm::Value *initializerValue = initExpr->GetValue(ctx);
        if (initializerValue)
            ctx->StoreInst(initializerValue, lvalue);
        return;
    }

    // There are two cases for initializing structs, arrays and vectors;
    // either a single initializer may be provided (float foo[3] = 0;), in
    // which case all of the elements are initialized to the given value,
    // or an initializer list may be provided (float foo[3] = { 1,2,3 }),
    // in which case the elements are initialized with the corresponding
    // values.
    const CollectionType *collectionType = 
        dynamic_cast<const CollectionType *>(type);
    if (collectionType != NULL) {
        std::string name;
        if (dynamic_cast<const StructType *>(type) != NULL)
            name = "struct";
        else if (dynamic_cast<const ArrayType *>(type) != NULL) 
            name = "array";
        else if (dynamic_cast<const VectorType *>(type) != NULL) 
            name = "vector";
        else 
            FATAL("Unexpected CollectionType in lInitSymbol()");

        ExprList *exprList = dynamic_cast<ExprList *>(initExpr);
        if (exprList != NULL) {
            // The { ... } case; make sure we have the same number of
            // expressions in the ExprList as we have struct members
            int nInits = exprList->exprs.size();
            if (nInits != collectionType->GetElementCount()) {
                Error(initExpr->pos, "Initializer for %s \"%s\" requires "
                      "%d values; %d provided.", name.c_str(), symName, 
                      collectionType->GetElementCount(), nInits);
                return;
            }

            // Initialize each element with the corresponding value from
            // the ExprList
            for (int i = 0; i < nInits; ++i) {
                llvm::Value *ep = ctx->GetElementPtrInst(lvalue, 0, i, "element");
                lInitSymbol(ep, symName, collectionType->GetElementType(i), 
                            exprList->exprs[i], ctx, pos);
            }
        }
        else
            Error(initExpr->pos, "Can't assign type \"%s\" to \"%s\".",
                  initExpr->GetType()->GetString().c_str(),
                  collectionType->GetString().c_str());
        return;
    }

    FATAL("Unexpected Type in lInitSymbol()");
}


void
DeclStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock()) 
        return;

    for (unsigned int i = 0; i < declaration->declarators.size(); ++i) {
        Declarator *decl = declaration->declarators[i];
        if (!decl || decl->isFunction) 
            continue;

        Symbol *sym = decl->sym;
        assert(decl->sym != NULL);
        const Type *type = sym->type;
        if (!type)
            continue;

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
        const ArrayType *at = dynamic_cast<const ArrayType *>(type);
        if (at && at->GetElementCount() == 0) {
            ExprList *exprList = dynamic_cast<ExprList *>(decl->initExpr);
            if (exprList) {
                ArrayType *t = at->GetSizedArray(exprList->exprs.size());
                assert(t != NULL);
                sym->type = type = t;
            }
            else {
                Error(sym->pos, "Can't declare an unsized array as a local "
                      "variable without providing an initializer expression to "
                      "set its size.");
                continue;
            }
        }

        // References must have initializer expressions as well.
        if (dynamic_cast<const ReferenceType *>(type) && decl->initExpr == NULL) {
            Error(sym->pos,
                  "Must provide initializer for reference-type variable \"%s\".",
                  sym->name.c_str());
            continue;
        }

        LLVM_TYPE_CONST llvm::Type *llvmType = type->LLVMType(g->ctx);
        assert(llvmType != NULL);

        if (declaration->declSpecs->storageClass == SC_STATIC) {
            // For static variables, we need a compile-time constant value
            // for its initializer; if there's no initializer, we use a
            // zero value.
            llvm::Constant *cinit = NULL;
            if (decl->initExpr) {
                cinit = decl->initExpr->GetConstant(type);
                if (!cinit)
                    Error(sym->pos, "Initializer for static variable \"%s\" must be a constant.",
                          sym->name.c_str());
            }
            if (!cinit)
                cinit = llvm::Constant::getNullValue(llvmType);

            // Allocate space for the static variable in global scope, so
            // that it persists across function calls
            sym->storagePtr =
                new llvm::GlobalVariable(*m->module, llvmType, type->IsConstType(),
                                         llvm::GlobalValue::InternalLinkage, cinit,
                                         llvm::Twine("static.") +
                                         llvm::Twine(sym->pos.first_line) + 
                                         llvm::Twine(".") + sym->name.c_str());
        }
        else {
            // For non-static variables, allocate storage on the stack
            sym->storagePtr = ctx->AllocaInst(llvmType, sym->name.c_str());
            // And then get it initialized...
            lInitSymbol(sym->storagePtr, sym->name.c_str(), type, decl->initExpr,
                        ctx, sym->pos);
        }

        // Finally, tell the FunctionEmitContext about the variable 
        ctx->EmitVariableDebugInfo(sym);
    }
}


Stmt *
DeclStmt::Optimize() {
    for (unsigned int i = 0; i < declaration->declarators.size(); ++i) {
        Declarator *decl = declaration->declarators[i];
        if (decl && decl->initExpr) {
            decl->initExpr = decl->initExpr->Optimize();

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
            Symbol *sym = decl->sym;
            if (sym->type && sym->type->IsConstType() && decl->initExpr && 
                dynamic_cast<ExprList *>(decl->initExpr) == NULL &&
                Type::Equal(decl->initExpr->GetType(), sym->type))
                sym->constValue = dynamic_cast<ConstExpr *>(decl->initExpr);
        }
    }
    return this;
}


Stmt *
DeclStmt::TypeCheck() {
    bool encounteredError = false;
    for (unsigned int i = 0; i < declaration->declarators.size(); ++i) {
        Declarator *decl = declaration->declarators[i];
        if (!decl) {
            encounteredError = true;
            continue;
        }

        if (!decl->initExpr)
            continue;
        decl->initExpr = decl->initExpr->TypeCheck();
        if (!decl->initExpr)
            continue;

        // get the right type for stuff like const float foo = 2; so that
        // the int->float type conversion is in there and we don't return
        // an int as the constValue later...
        const Type *type = decl->sym->type;
        if (dynamic_cast<const AtomicType *>(type) != NULL ||
            dynamic_cast<const EnumType *>(type) != NULL) {
            // If it's an expr list with an atomic type, we'll later issue
            // an error.  Need to leave decl->initExpr as is in that case so it
            // is in fact caught later, though.
            if (dynamic_cast<ExprList *>(decl->initExpr) == NULL)
                decl->initExpr = decl->initExpr->TypeConv(type, "initializer");
        }
    }
    return encounteredError ? NULL : this;
}


void
DeclStmt::Print(int indent) const {
    printf("%*cDecl Stmt:", indent, ' ');
    pos.Print();
    if (declaration) 
        declaration->Print();
    printf("\n");
}


///////////////////////////////////////////////////////////////////////////
// IfStmt

IfStmt::IfStmt(Expr *t, Stmt *ts, Stmt *fs, bool doUnif, SourcePos p) 
    : Stmt(p), test(t), trueStmts(ts), falseStmts(fs), 
      doCoherentCheck(doUnif && !g->opt.disableCoherentControlFlow) {
}


static void
lEmitIfStatements(FunctionEmitContext *ctx, Stmt *stmts, const char *trueOrFalse) {
    if (!stmts)
        return;

    if (dynamic_cast<StmtList *>(stmts) == NULL)
        ctx->StartScope();
    ctx->AddInstrumentationPoint(trueOrFalse);
    stmts->EmitCode(ctx);
    if (dynamic_cast<const StmtList *>(stmts) == NULL)
        ctx->EndScope();
}

void
IfStmt::EmitCode(FunctionEmitContext *ctx) const {
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
    if (isUniform) {
        ctx->StartUniformIf(ctx->GetMask());
        if (doCoherentCheck)
            Warning(test->pos, "Uniform condition supplied to cif statement.");

        // 'If' statements with uniform conditions are relatively
        // straightforward.  We evaluate the condition and then jump to
        // either the 'then' or 'else' clause depending on its value.
        llvm::Value *vtest = test->GetValue(ctx);
        if (vtest != NULL) {
            llvm::BasicBlock *bthen = ctx->CreateBasicBlock("if_then");
            llvm::BasicBlock *belse = ctx->CreateBasicBlock("if_else");
            llvm::BasicBlock *bexit = ctx->CreateBasicBlock("if_exit");

            // Jump to the appropriate basic block based on the value of
            // the 'if' test
            ctx->BranchInst(bthen, belse, vtest);

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
        }
        ctx->EndIf();
    }
    else {
        // Code for 'If' statemnts with 'varying' conditions can be
        // generated in two ways; one takes some care to see if all of the
        // active program instances want to follow only the 'true' or
        // 'false' cases, and the other always runs both cases but sets the
        // mask appropriately.  The first case is handled by the
        // IfStmt::emitCoherentTests() call, and the second is handled by
        // IfStmt::emitMaskedTrueAndFalse().
        llvm::Value *testValue = test->GetValue(ctx);
        if (testValue) {
            if (doCoherentCheck) 
                emitCoherentTests(ctx, testValue);
            else {
                llvm::Value *oldMask = ctx->GetMask();
                ctx->StartVaryingIf(oldMask);
                emitMaskedTrueAndFalse(ctx, oldMask, testValue);
                ctx->EndIf();
            }
        }
    }
}


Stmt *
IfStmt::Optimize() {
    if (test) 
        test = test->Optimize();
    if (trueStmts) 
        trueStmts = trueStmts->Optimize();
    if (falseStmts) 
        falseStmts = falseStmts->Optimize();
    return this;
}


Stmt *IfStmt::TypeCheck() {
    if (test) {
        test = test->TypeCheck();
        if (test) {
            const Type *testType = test->GetType();
            if (testType) {
                bool isUniform = testType->IsUniformType() && !g->opt.disableUniformControlFlow;
                if (!testType->IsNumericType() && !testType->IsBoolType()) {
                    Error(test->pos, "Type \"%s\" can't be converted to boolean for \"if\" test.",
                          testType->GetString().c_str());
                    return NULL;
                }
                test = new TypeCastExpr(isUniform ? AtomicType::UniformBool : 
                                        AtomicType::VaryingBool, 
                                        test, test->pos);
                assert(test);
            }
        }
    }
    if (trueStmts)
        trueStmts = trueStmts->TypeCheck();
    if (falseStmts) 
        falseStmts = falseStmts->TypeCheck();

    return this;
}


void
IfStmt::Print(int indent) const {
    printf("%*cIf Stmt %s", indent, ' ', doCoherentCheck ? "DO COHERENT CHECK" : "");
    pos.Print();
    printf("\n%*cTest: ", indent+4, ' ');
    test->Print();
    printf("\n");
    if (trueStmts) {
        printf("%*cTrue:\n", indent+4, ' ');
        trueStmts->Print(indent+8);
    }
    if (falseStmts) {
        printf("%*cFalse:\n", indent+4, ' ');
        falseStmts->Print(indent+8);
    }
}


/** Emit code to run both the true and false statements for the if test,
    with the mask set appropriately before runnign each one. 
*/
void
IfStmt::emitMaskedTrueAndFalse(FunctionEmitContext *ctx, llvm::Value *oldMask, 
                               llvm::Value *test) const {
    if (trueStmts) {
        ctx->MaskAnd(oldMask, test);
        lEmitIfStatements(ctx, trueStmts, "if: expr mixed, true statements");
        // under varying control flow,, returns can't stop instruction
        // emission, so this better be non-NULL...
        assert(ctx->GetCurrentBasicBlock()); 
    }
    if (falseStmts) {
        ctx->MaskAndNot(oldMask, test);
        lEmitIfStatements(ctx, falseStmts, "if: expr mixed, false statements");
        assert(ctx->GetCurrentBasicBlock());
    }
}


/** Emit code for an if test that checks the mask and the test values and
    tries to be smart about jumping over code that doesn't need to be run.
 */
void
IfStmt::emitCoherentTests(FunctionEmitContext *ctx, llvm::Value *ltest) const {
    llvm::Value *oldMask = ctx->GetMask();
    if (oldMask == LLVMMaskAllOn) {
        // We can tell that the mask is on statically at compile time; just
        // emit code for the 'if test with the mask all on' path
        llvm::BasicBlock *bDone = ctx->CreateBasicBlock("cif_done");
        emitMaskAllOn(ctx, ltest, bDone);
        ctx->SetCurrentBasicBlock(bDone);
    }
    else {
        // We can't tell if the mask going into the if is all on at the
        // compile time.  Emit code to check for this and then either run
        // the code for the 'all on' or the 'mixed' case depending on the
        // mask's value at runtime.
        llvm::BasicBlock *bAllOn = ctx->CreateBasicBlock("cif_mask_all");
        llvm::BasicBlock *bMixedOn = ctx->CreateBasicBlock("cif_mask_mixed");
        llvm::BasicBlock *bDone = ctx->CreateBasicBlock("cif_done");

        // Jump to either bAllOn or bMixedOn, depending on the mask's value 
        llvm::Value *maskAllQ = ctx->All(oldMask);
        ctx->BranchInst(bAllOn, bMixedOn, maskAllQ);

        // Emit code for the 'mask all on' case
        ctx->SetCurrentBasicBlock(bAllOn);
        // We start by explicitly storing "all on" into the mask mask.
        // Note that this doesn't change its actual value, but doing so
        // lets the compiler see what's going on so that subsequent
        // optimizations for code emitted here can operate with the
        // knowledge that the mask is definitely all on (until it modifies
        // the mask itself).
        ctx->SetMask(LLVMMaskAllOn);
        emitMaskAllOn(ctx, ltest, bDone);
        
        // And emit code for the mixed mask case
        ctx->SetCurrentBasicBlock(bMixedOn);
        emitMaskMixed(ctx, oldMask, ltest, bDone);

        // When done, set the current basic block to the block that the two
        // paths above jump to when they're done.
        ctx->SetCurrentBasicBlock(bDone);
    }
}


/** Emits code for 'if' tests under the case where we know that the program
    mask is all on.
 */
void
IfStmt::emitMaskAllOn(FunctionEmitContext *ctx, llvm::Value *ltest, 
                      llvm::BasicBlock *bDone) const {
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
    assert(ctx->GetCurrentBasicBlock());
    ctx->EndIf();
    ctx->BranchInst(bDone);
}


/** Emits code that checks to see if for all of the lanes where the mask is
    on, the test has the value true.
 */
static llvm::Value *
lTestMatchesMask(FunctionEmitContext *ctx, llvm::Value *test, llvm::Value *mask) {
    llvm::Value *testAndMask = ctx->BinaryOperator(llvm::Instruction::And, test,
                                                   mask, "test&mask");
    return ctx->MasksAllEqual(testAndMask, mask);
}


/** Emit code for an 'if' test where the lane mask is known to be mixed
    on/off going into it.
 */
void
IfStmt::emitMaskMixed(FunctionEmitContext *ctx, llvm::Value *oldMask, 
                      llvm::Value *ltest, llvm::BasicBlock *bDone) const {
    // First, see if, for all of the lanes where the mask is on, if the
    // value of the test is on.  (i.e. (test&mask) == mask).  In this case,
    // we only need to run the 'true' case code, since the lanes where the
    // test was false aren't supposed to be running here anyway.
     llvm::Value *testAllEqual = lTestMatchesMask(ctx, ltest, oldMask);
    llvm::BasicBlock *bTestAll = ctx->CreateBasicBlock("cif_mixed_test_all");
    llvm::BasicBlock *bTestAnyCheck = ctx->CreateBasicBlock("cif_mixed_test_any_check");
    ctx->BranchInst(bTestAll, bTestAnyCheck, testAllEqual);

    // Emit code for the (test&mask)==mask case.  Not only do we only need
    // to emit code for the true statements, but we don't need to modify
    // the mask's value; it's already correct.
    ctx->SetCurrentBasicBlock(bTestAll);
    ctx->StartVaryingIf(ctx->GetMask());
    lEmitIfStatements(ctx, trueStmts, "cif: all running lanes want just true stmts");
    assert(ctx->GetCurrentBasicBlock());
    ctx->EndIf();
    ctx->BranchInst(bDone);

    // Next, see if the active lanes only need to run the false case--i.e. if
    // (~test & mask) == mask.
    ctx->SetCurrentBasicBlock(bTestAnyCheck);
    llvm::Value *notTest = ctx->BinaryOperator(llvm::Instruction::Xor, LLVMMaskAllOn,
                                               ltest, "~test");
    llvm::Value *notMatchesMask = lTestMatchesMask(ctx, notTest, oldMask);
    llvm::BasicBlock *bTestAllNot = ctx->CreateBasicBlock("cif_mixed_test_none");
    llvm::BasicBlock *bTestMixed = ctx->CreateBasicBlock("cif_mixed_test_mixed");
    ctx->BranchInst(bTestAllNot, bTestMixed, notMatchesMask);

    // Emit code for the (~test & mask) == mask case.  We only need the
    // 'false' statements and again don't need to modify the value of the
    // mask.
    ctx->SetCurrentBasicBlock(bTestAllNot);
    ctx->StartVaryingIf(ctx->GetMask());
    lEmitIfStatements(ctx, falseStmts, "cif: all running lanes want just false stmts");
    assert(ctx->GetCurrentBasicBlock());
    ctx->EndIf();
    ctx->BranchInst(bDone);

    // It's mixed; we need to run both the true and false cases and also do
    // mask update stuff.
    ctx->SetCurrentBasicBlock(bTestMixed);
    ctx->StartVaryingIf(ctx->GetMask());
    emitMaskedTrueAndFalse(ctx, oldMask, ltest);
    ctx->EndIf();
    ctx->BranchInst(bDone);
}


///////////////////////////////////////////////////////////////////////////
// DoStmt

/** Given a statment, walk through it to see if there is a 'break' or
    'continue' statement inside if its children, under varying control
    flow.  We need to detect this case for loops since what might otherwise
    look like a 'uniform' loop needs to have code emitted to do all of the
    lane management stuff if this is the case.
 */ 
static bool
lHasVaryingBreakOrContinue(Stmt *stmt, bool inVaryingCF = false) {
    StmtList *sl;
    IfStmt *is;

    if ((sl = dynamic_cast<StmtList *>(stmt)) != NULL) {
        // Recurse through the children statements 
        const std::vector<Stmt *> &stmts = sl->GetStatements();
        for (unsigned int i = 0; i < stmts.size(); ++i)
            if (lHasVaryingBreakOrContinue(stmts[i], inVaryingCF))
                return true;
    }
    else if ((is = dynamic_cast<IfStmt *>(stmt)) != NULL) {
        // We've come to an 'if'.  Is the test type varying?  If so, then
        // we're under 'varying' control flow when we recurse through the
        // true and false statements.
        if (is->test != NULL) {
            const Type *type = is->test->GetType();
            if (type)
                inVaryingCF |= type->IsVaryingType();
        }

        if (lHasVaryingBreakOrContinue(is->trueStmts, inVaryingCF) ||
            lHasVaryingBreakOrContinue(is->falseStmts, inVaryingCF))
            return true;
    }
    else if (dynamic_cast<BreakStmt *>(stmt) != NULL) {
        if (inVaryingCF)
            return true;
    }
    else if (dynamic_cast<ContinueStmt *>(stmt) != NULL) {
        if (inVaryingCF)
            return true;
    }
    // Important: note that we don't recurse into do/for loops here but
    // just return false.  For the question of whether a given loop needs
    // to do mask management stuff, breaks/continues inside nested loops
    // inside of them don't matter.
    return false;
}


DoStmt::DoStmt(Expr *t, Stmt *s, bool cc, SourcePos p) 
    : Stmt(p), testExpr(t), bodyStmts(s), 
      doCoherentCheck(cc && !g->opt.disableCoherentControlFlow) {
}


void DoStmt::EmitCode(FunctionEmitContext *ctx) const {
    // Check for things that could be NULL due to earlier errors during
    // compilation.
    if (!ctx->GetCurrentBasicBlock()) 
        return;
    if (!testExpr || !testExpr->GetType()) 
        return;

    bool uniformTest = testExpr->GetType()->IsUniformType();
    if (uniformTest && doCoherentCheck)
        Warning(pos, "Uniform condition supplied to \"cdo\" statement.");

    llvm::BasicBlock *bloop = ctx->CreateBasicBlock("do_loop");
    llvm::BasicBlock *bexit = ctx->CreateBasicBlock("do_exit");
    llvm::BasicBlock *btest = ctx->CreateBasicBlock("do_test");

    ctx->StartLoop(bexit, btest, uniformTest, ctx->GetMask());

    // Start by jumping into the loop body
    ctx->BranchInst(bloop);

    // And now emit code for the loop body
    ctx->SetCurrentBasicBlock(bloop);
    ctx->SetLoopMask(ctx->GetMask());
    ctx->SetDebugPos(pos);
    // FIXME: in the StmtList::EmitCode() method takes starts/stops a new
    // scope around the statements in the list.  So if the body is just a
    // single statement (and thus not a statement list), we need a new
    // scope, but we don't want two scopes in the StmtList case.
    if (!dynamic_cast<StmtList *>(bodyStmts))
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
        ctx->SetMask(LLVMMaskAllOn);
        if (bodyStmts)
            bodyStmts->EmitCode(ctx);
        assert(ctx->GetCurrentBasicBlock());
        ctx->BranchInst(btest);

        // The mask is mixed.  Just emit the code for the loop body.
        ctx->SetCurrentBasicBlock(bMixed);
        if (bodyStmts)
            bodyStmts->EmitCode(ctx);
        assert(ctx->GetCurrentBasicBlock());
        ctx->BranchInst(btest);
    }
    else {
        // Otherwise just emit the code for the loop body.  The current
        // mask is good.
        if (bodyStmts)
            bodyStmts->EmitCode(ctx);
        if (ctx->GetCurrentBasicBlock())
            ctx->BranchInst(btest);
    }
    // End the scope we started above, if needed.
    if (!dynamic_cast<StmtList *>(bodyStmts))
        ctx->EndScope();

    // Now emit code for the loop test.
    ctx->SetCurrentBasicBlock(btest);
    // First, emit code to restore the mask value for any lanes that
    // executed a 'continue' during the current loop before we go and emit
    // the code for the test.  This is only necessary for varying loops;
    // 'uniform' loops just jump when they hit a continue statement and
    // don't mess with the mask.
    if (!uniformTest)
        ctx->RestoreContinuedLanes();
    llvm::Value *testValue = testExpr->GetValue(ctx);
    if (!testValue)
        return;

    if (uniformTest)
        // For the uniform case, just jump to the top of the loop or the
        // exit basic block depending on the value of the test.
        ctx->BranchInst(bloop, bexit, testValue);
    else {
        // For the varying case, update the mask based on the value of the
        // test.  If any program instances still want to be running, jump
        // to the top of the loop.  Otherwise, jump out.
        llvm::Value *mask = ctx->GetMask();
        ctx->MaskAnd(mask, testValue);
        ctx->BranchIfMaskAny(bloop, bexit);
    }

    // ...and we're done.  Set things up for subsequent code to be emitted
    // in the right basic block.
    ctx->SetCurrentBasicBlock(bexit);
    ctx->EndLoop();
}


Stmt *
DoStmt::Optimize() {
    if (testExpr) 
        testExpr = testExpr->Optimize();
    if (bodyStmts) 
        bodyStmts = bodyStmts->Optimize();
    return this;
}


Stmt *
DoStmt::TypeCheck() {
    if (testExpr) {
        testExpr = testExpr->TypeCheck();
        if (testExpr) {
            const Type *testType = testExpr->GetType();
            if (testType) {
                if (!testType->IsNumericType() && !testType->IsBoolType()) {
                    Error(testExpr->pos, "Type \"%s\" can't be converted to boolean for \"while\" "
                          "test in \"do\" loop.", testExpr->GetType()->GetString().c_str());
                    return NULL;
                }

                // Should the test condition for the loop be uniform or
                // varying?  It can be uniform only if three conditions are
                // met.  First and foremost, the type of the test condition
                // must be uniform.  Second, the user must not have set the
                // dis-optimization option that disables uniform flow
                // control.
                //
                // Thirdly, and most subtlely, there must not be any break
                // or continue statements inside the loop that are within
                // the scope of a 'varying' if statement.  If there are,
                // then we type cast the test to be 'varying', so that the
                // code generated for the loop includes masking stuff, so
                // that we can track which lanes actually want to be
                // running, accounting for breaks/continues.
                bool uniformTest = (testType->IsUniformType() &&
                                    !g->opt.disableUniformControlFlow &&
                                    !lHasVaryingBreakOrContinue(bodyStmts));
                testExpr = new TypeCastExpr(uniformTest ? AtomicType::UniformBool :
                                                          AtomicType::VaryingBool,
                                            testExpr, testExpr->pos);
            }
        }
    }

    if (bodyStmts) 
        bodyStmts = bodyStmts->TypeCheck();
    return this;
}


void
DoStmt::Print(int indent) const {
    printf("%*cDo Stmt", indent, ' ');
    pos.Print();
    printf(":\n");
    printf("%*cTest: ", indent+4, ' ');
    if (testExpr) testExpr->Print();
    printf("\n");
    if (bodyStmts) {
        printf("%*cStmts:\n", indent+4, ' ');
        bodyStmts->Print(indent+8);
    }
}


///////////////////////////////////////////////////////////////////////////
// ForStmt

ForStmt::ForStmt(Stmt *i, Expr *t, Stmt *s, Stmt *st, bool cc, SourcePos p) 
    : Stmt(p), init(i), test(t), step(s), stmts(st), 
      doCoherentCheck(cc && !g->opt.disableCoherentControlFlow) {
}


void
ForStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock()) 
        return;

    llvm::BasicBlock *btest = ctx->CreateBasicBlock("for_test");
    llvm::BasicBlock *bstep = ctx->CreateBasicBlock("for_step");
    llvm::BasicBlock *bloop = ctx->CreateBasicBlock("for_loop");
    llvm::BasicBlock *bexit = ctx->CreateBasicBlock("for_exit");

    bool uniformTest = test ? test->GetType()->IsUniformType() :
        (!g->opt.disableUniformControlFlow &&
         !lHasVaryingBreakOrContinue(stmts));

    ctx->StartLoop(bexit, bstep, uniformTest, ctx->GetMask());
    ctx->SetDebugPos(pos);

    // If we have an initiailizer statement, start by emitting the code for
    // it and then jump into the loop test code.  (Also start a new scope
    // since the initiailizer may be a declaration statement).
    if (init) {
        assert(dynamic_cast<StmtList *>(init) == NULL);
        ctx->StartScope();
        init->EmitCode(ctx);
    }
    ctx->BranchInst(btest);

    assert(ctx->GetCurrentBasicBlock());
#if 0
    if (!ctx->GetCurrentBasicBlock()) {
        // when does this happen??
        if (init)
            ctx->EndScope();
        ctx->EndLoop();
        return;
    }
#endif

    // Emit code to get the value of the loop test.  If no test expression
    // was provided, just go with a true value.
    ctx->SetCurrentBasicBlock(btest);
    llvm::Value *ltest = NULL;
    if (test) {
        ltest = test->GetValue(ctx);
        if (!ltest) {
            ctx->EndScope();
            ctx->EndLoop();
            return;
        }
    }
    else
        ltest = uniformTest ? LLVMTrue : LLVMBoolVector(true);

    // Now use the test's value.  For a uniform loop, we can either jump to
    // the loop body or the loop exit, based on whether it's true or false.
    // For a non-uniform loop, we update the mask and jump into the loop if
    // any of the mask values are true.
    if (uniformTest) {
        if (doCoherentCheck)
            Warning(pos, "Uniform condition supplied to cfor/cwhile statement.");
        assert(ltest->getType() == LLVMTypes::BoolType);
        ctx->BranchInst(bloop, bexit, ltest);
    }
    else {
        llvm::Value *mask = ctx->GetMask();
        ctx->MaskAnd(mask, ltest);
        ctx->BranchIfMaskAny(bloop, bexit);
    }

    // On to emitting the code for the loop body.
    ctx->SetCurrentBasicBlock(bloop);
    ctx->SetLoopMask(ctx->GetMask());
    ctx->AddInstrumentationPoint("for loop body");
    if (!dynamic_cast<StmtList *>(stmts))
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
        ctx->SetMask(LLVMMaskAllOn);
        if (stmts)
            stmts->EmitCode(ctx);
        assert(ctx->GetCurrentBasicBlock());
        ctx->BranchInst(bstep);

        // Emit code for the mask being mixed.  We should never run the
        // loop with the mask all off, based on the BranchIfMaskAny call
        // above.
        ctx->SetCurrentBasicBlock(bMixed);
        if (stmts)
            stmts->EmitCode(ctx);
        ctx->BranchInst(bstep);
    }
    else {
        // For both uniform loops and varying loops without the coherence
        // check, we know that at least one program instance wants to be
        // running the loop, so just emit code for the loop body and jump
        // to the loop step code.
        if (stmts)
            stmts->EmitCode(ctx);
        if (ctx->GetCurrentBasicBlock())
            ctx->BranchInst(bstep);
    }
    if (!dynamic_cast<StmtList *>(stmts))
        ctx->EndScope();

    // Emit code for the loop step.  First, restore the lane mask of any
    // program instances that executed a 'continue' during the previous
    // iteration.  Then emit code for the loop step and then jump to the
    // test code.
    ctx->SetCurrentBasicBlock(bstep);
    ctx->RestoreContinuedLanes();
    if (step)
        step->EmitCode(ctx);
    ctx->BranchInst(btest);

    // Set the current emission basic block to the loop exit basic block
    ctx->SetCurrentBasicBlock(bexit);
    if (init)
        ctx->EndScope();
    ctx->EndLoop();
}


Stmt *
ForStmt::Optimize() {
    if (test) 
        test = test->Optimize();
    if (init) 
        init = init->Optimize();
    if (step) 
        step = step->Optimize();
    if (stmts) 
        stmts = stmts->Optimize();
    return this;
}


Stmt *
ForStmt::TypeCheck() {
    if (test) {
        test = test->TypeCheck();
        if (test) {
            const Type *testType = test->GetType();
            if (testType) {
                if (!testType->IsNumericType() && !testType->IsBoolType()) {
                    Error(test->pos, "Type \"%s\" can't be converted to boolean for for loop test.",
                          test->GetType()->GetString().c_str());
                    return NULL;
                }

                // See comments in DoStmt::TypeCheck() regarding
                // 'uniformTest' and the type cast here.
                bool uniformTest = (testType->IsUniformType() &&
                                    !g->opt.disableUniformControlFlow &&
                                    !lHasVaryingBreakOrContinue(stmts));
                test = new TypeCastExpr(uniformTest ? AtomicType::UniformBool :
                                                      AtomicType::VaryingBool,
                                        test, test->pos);
            }
        }
    }

    if (init) 
        init = init->TypeCheck();
    if (step) 
        step = step->TypeCheck();
    if (stmts) 
        stmts = stmts->TypeCheck();
    return this;
}


void
ForStmt::Print(int indent) const {
    printf("%*cFor Stmt", indent, ' ');
    pos.Print();
    printf("\n");
    if (init) {
        printf("%*cInit:\n", indent+4, ' ');
        init->Print(indent+8);
    }
    if (test) {
        printf("%*cTest: ", indent+4, ' ');
        test->Print();
        printf("\n");
    }
    if (step) {
        printf("%*cStep:\n", indent+4, ' ');
        step->Print(indent+8);
    }
    if (stmts) {
        printf("%*cStmts:\n", indent+4, ' ');
        stmts->Print(indent+8);
    }
}

///////////////////////////////////////////////////////////////////////////
// BreakStmt

BreakStmt::BreakStmt(bool cc, SourcePos p) 
    : Stmt(p), doCoherenceCheck(cc && !g->opt.disableCoherentControlFlow) {
}


void
BreakStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock()) 
        return;

    ctx->SetDebugPos(pos);
    ctx->Break(doCoherenceCheck);
}


Stmt *
BreakStmt::Optimize() {
    return this;
}


Stmt *
BreakStmt::TypeCheck() {
    return this;
}


void
BreakStmt::Print(int indent) const {
    printf("%*c%sBreak Stmt", indent, ' ', doCoherenceCheck ? "Coherent " : "");
    pos.Print();
    printf("\n");
}


///////////////////////////////////////////////////////////////////////////
// ContinueStmt

ContinueStmt::ContinueStmt(bool cc, SourcePos p) 
    : Stmt(p), doCoherenceCheck(cc && !g->opt.disableCoherentControlFlow) {
}


void
ContinueStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock()) 
        return;

    ctx->SetDebugPos(pos);
    ctx->Continue(doCoherenceCheck);
}


Stmt *
ContinueStmt::Optimize() {
    return this;
}


Stmt *
ContinueStmt::TypeCheck() {
    return this;
}


void
ContinueStmt::Print(int indent) const {
    printf("%*c%sContinue Stmt", indent, ' ', doCoherenceCheck ? "Coherent " : "");
    pos.Print();
    printf("\n");
}


///////////////////////////////////////////////////////////////////////////
// ReturnStmt

ReturnStmt::ReturnStmt(Expr *v, bool cc, SourcePos p) 
    : Stmt(p), val(v), 
      doCoherenceCheck(cc && !g->opt.disableCoherentControlFlow) {
}


void
ReturnStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock()) 
        return;

    ctx->SetDebugPos(pos);
    ctx->CurrentLanesReturned(val, doCoherenceCheck);
}


Stmt *
ReturnStmt::Optimize() {
    if (val) 
        val = val->Optimize();
    return this;
}


Stmt *
ReturnStmt::TypeCheck() {
    // FIXME: We don't have ctx->functionType available here; should we?
    // We ned up needing to type conversion stuff in EmitCode() method via
    // FunctionEmitContext::SetReturnValue as a result, which is kind of ugly...
    if (val)
        val = val->TypeCheck();
    return this;
}


void
ReturnStmt::Print(int indent) const {
    printf("%*c%sReturn Stmt", indent, ' ', doCoherenceCheck ? "Coherent " : "");
    pos.Print();
    if (val) val->Print();
    else printf("(void)");
    printf("\n");
}


///////////////////////////////////////////////////////////////////////////
// StmtList

void
StmtList::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock()) 
        return;

    ctx->StartScope();
    ctx->SetDebugPos(pos);
    for (unsigned int i = 0; i < stmts.size(); ++i)
        if (stmts[i])
            stmts[i]->EmitCode(ctx);
    ctx->EndScope();
}


Stmt *
StmtList::Optimize() {
    for (unsigned int i = 0; i < stmts.size(); ++i)
        if (stmts[i])
            stmts[i] = stmts[i]->Optimize();
    return this;
}


Stmt *
StmtList::TypeCheck() {
    for (unsigned int i = 0; i < stmts.size(); ++i)
        if (stmts[i])
            stmts[i] = stmts[i]->TypeCheck();
    return this;
}


void
StmtList::Print(int indent) const {
    printf("%*cStmt List", indent, ' ');
    pos.Print();
    printf(":\n");
    for (unsigned int i = 0; i < stmts.size(); ++i)
        if (stmts[i])
            stmts[i]->Print(indent+4);
}


///////////////////////////////////////////////////////////////////////////
// PrintStmt

PrintStmt::PrintStmt(const std::string &f, Expr *v, SourcePos p) 
    : Stmt(p), format(f), values(v) {
}

/* Because the pointers to values that are passed to __do_print() are all
   void *s (and because ispc print() formatting strings statements don't
   encode types), we pass along a string to __do_print() where the i'th
   character encodes the type of the i'th value to be printed.  Needless to
   say, the encoding chosen here and the decoding code in __do_print() need
   to agree on the below!
 */
static char
lEncodeType(const Type *t) {
    if (t == AtomicType::UniformBool)   return 'b';
    if (t == AtomicType::VaryingBool)   return 'B';
    if (t == AtomicType::UniformInt32)  return 'i';
    if (t == AtomicType::VaryingInt32)  return 'I';
    if (t == AtomicType::UniformUInt32) return 'u';
    if (t == AtomicType::VaryingUInt32) return 'U';
    if (t == AtomicType::UniformFloat)  return 'f';
    if (t == AtomicType::VaryingFloat)  return 'F';
    if (t == AtomicType::UniformInt64)  return 'l';
    if (t == AtomicType::VaryingInt64)  return 'L';
    if (t == AtomicType::UniformUInt64) return 'v';
    if (t == AtomicType::VaryingUInt64) return 'V';
    if (t == AtomicType::UniformDouble) return 'd';
    if (t == AtomicType::VaryingDouble) return 'D';
    else return '\0';
}


/** Given an Expr for a value to be printed, emit the code to evaluate the
    expression and store the result to alloca'd memory.  Update the
    argTypes string with the type encoding for this expression.
 */
static llvm::Value *
lProcessPrintArg(Expr *expr, FunctionEmitContext *ctx, std::string &argTypes) {
    const Type *type = expr->GetType();
    if (type == NULL)
        return NULL;

    if (dynamic_cast<const ReferenceType *>(type) != NULL) {
        expr = new DereferenceExpr(expr, expr->pos);
        type = expr->GetType();
        if (type == NULL)
            return NULL;
    }

    char t = lEncodeType(type->GetAsNonConstType());
    if (t == '\0') {
        Error(expr->pos, "Only atomic types are allowed in print statements; "
              "type \"%s\" is illegal.", type->GetString().c_str());
        return NULL;
    }
    else {
        argTypes.push_back(t);

        LLVM_TYPE_CONST llvm::Type *llvmExprType = type->LLVMType(g->ctx);
        llvm::Value *ptr = ctx->AllocaInst(llvmExprType, "print_arg");
        llvm::Value *val = expr->GetValue(ctx);
        if (!val)
            return NULL;
        ctx->StoreInst(val, ptr);

        ptr = ctx->BitCastInst(ptr, LLVMTypes::VoidPointerType);
        return ptr;
    }
}


/* PrintStmt works closely with the __do_print() function implemented in
   the stdlib-c.c file.  In particular, the EmitCode() method here needs to
   take the arguments passed to it from ispc and generate a valid call to
   __do_print() with the information that __do_print() then needs to do the
   actual printing work at runtime.
 */
void
PrintStmt::EmitCode(FunctionEmitContext *ctx) const {
    ctx->SetDebugPos(pos);

    // __do_print takes 5 arguments; we'll get them stored in the args[] array
    // in the code emitted below
    //
    // 1. the format string
    // 2. a string encoding the types of the values being printed, 
    //    one character per value
    // 3. the number of running program instances (i.e. the target's
    //    vector width)
    // 4. the current lane mask
    // 5. a pointer to an array of pointers to the values to be printed
    llvm::Value *args[5];
    std::string argTypes;

    if (values == NULL)
        args[4] = NULL;
    else {
        // Get the values passed to the print() statement evaluated and
        // stored in memory so that we set up the array of pointers to them
        // for the 5th __do_print() argument
        ExprList *elist = dynamic_cast<ExprList *>(values);
        int nArgs = elist ? elist->exprs.size() : 1;

        // Allocate space for the array of pointers to values to be printed 
        LLVM_TYPE_CONST llvm::Type *argPtrArrayType = 
            llvm::ArrayType::get(LLVMTypes::VoidPointerType, nArgs);
        llvm::Value *argPtrArray = ctx->AllocaInst(argPtrArrayType,
                                                   "print_arg_ptrs");
        // Store the array pointer as a void **, which is what __do_print()
        // expects
        args[4] = ctx->BitCastInst(argPtrArray, 
                                   llvm::PointerType::get(LLVMTypes::VoidPointerType, 0));

        // Now, for each of the arguments, emit code to evaluate its value
        // and store the value into alloca'd storage.  Then store the
        // pointer to the alloca'd storage into argPtrArray.
        if (elist) {
            for (unsigned int i = 0; i < elist->exprs.size(); ++i) {
                Expr *expr = elist->exprs[i];
                if (!expr)
                    return;
                llvm::Value *ptr = lProcessPrintArg(expr, ctx, argTypes);
                if (!ptr)
                    return;

                llvm::Value *arrayPtr = ctx->GetElementPtrInst(argPtrArray, 0, i);
                ctx->StoreInst(ptr, arrayPtr);
            }
        }
        else {
            llvm::Value *ptr = lProcessPrintArg(values, ctx, argTypes);
            if (!ptr)
                return;
            llvm::Value *arrayPtr = ctx->GetElementPtrInst(argPtrArray, 0, 0);
            ctx->StoreInst(ptr, arrayPtr);
        }
    }

    // Now we can emit code to call __do_print()
    llvm::Function *printFunc = m->module->getFunction("__do_print");
    assert(printFunc);

    // Set up the rest of the parameters to it
    args[0] = ctx->GetStringPtr(format);
    args[1] = ctx->GetStringPtr(argTypes);
    args[2] = LLVMInt32(g->target.vectorWidth);
    args[3] = ctx->LaneMask(ctx->GetMask());
    std::vector<llvm::Value *> argVec(&args[0], &args[5]);
    ctx->CallInst(printFunc, argVec, "");
}


void
PrintStmt::Print(int indent) const {
    printf("%*cPrint Stmt (%s)", indent, ' ', format.c_str());
}


Stmt *
PrintStmt::Optimize() {
    if (values) 
        values = values->Optimize();
    return this;
}


Stmt *
PrintStmt::TypeCheck() {
    if (values) 
        values = values->TypeCheck();
    return this;
}
