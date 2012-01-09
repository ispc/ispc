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
// Stmt

Stmt *
Stmt::Optimize() {
    return this;
}


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
ExprStmt::TypeCheck() {
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


int
ExprStmt::EstimateCost() const {
    return 0;
}


///////////////////////////////////////////////////////////////////////////
// DeclStmt

DeclStmt::DeclStmt(const std::vector<VariableDeclaration> &v, SourcePos p)
    : Stmt(p), vars(v) {
}


static bool
lPossiblyResolveFunctionOverloads(Expr *expr, const Type *type) {
    FunctionSymbolExpr *fse = NULL;
    const FunctionType *funcType = NULL;
    if (dynamic_cast<const PointerType *>(type) != NULL &&
        (funcType = dynamic_cast<const FunctionType *>(type->GetBaseType())) &&
        (fse = dynamic_cast<FunctionSymbolExpr *>(expr)) != NULL) {
        // We're initializing a function pointer with a function symbol,
        // which in turn may represent an overloaded function.  So we need
        // to try to resolve the overload based on the type of the symbol
        // we're initializing here.
        std::vector<const Type *> paramTypes;
        for (int i = 0; i < funcType->GetNumParameters(); ++i)
            paramTypes.push_back(funcType->GetParameterType(i));

        if (fse->ResolveOverloads(expr->pos, paramTypes) == false)
            return false;
    }
    return true;
}


/** Utility routine that emits code to initialize a symbol given an
    initializer expression.

    @param lvalue    Memory location of storage for the symbol's data
    @param symName   Name of symbol (used in error messages)
    @param symType   Type of variable being initialized
    @param initExpr  Expression for the initializer
    @param ctx       FunctionEmitContext to use for generating instructions
    @param pos       Source file position of the variable being initialized
*/
static void
lInitSymbol(llvm::Value *lvalue, const char *symName, const Type *symType,
            Expr *initExpr, FunctionEmitContext *ctx, SourcePos pos) {
    if (initExpr == NULL)
        // leave it uninitialized
        return;

    // If the initializer is a straight up expression that isn't an
    // ExprList, then we'll see if we can type convert it to the type of
    // the variable.
    if (dynamic_cast<ExprList *>(initExpr) == NULL) {
        if (lPossiblyResolveFunctionOverloads(initExpr, symType) == false)
            return;
        initExpr = TypeConvertExpr(initExpr, symType, "initializer");

        if (initExpr != NULL) {
            llvm::Value *initializerValue = initExpr->GetValue(ctx);
            if (initializerValue != NULL)
                // Bingo; store the value in the variable's storage
                ctx->StoreInst(initializerValue, lvalue);
            return;
        }
    }

    // Atomic types and enums can't be initialized with { ... } initializer
    // expressions, so print an error and return if that's what we've got
    // here..
    if (dynamic_cast<const AtomicType *>(symType) != NULL ||
        dynamic_cast<const EnumType *>(symType) != NULL ||
        dynamic_cast<const PointerType *>(symType) != NULL) {
        ExprList *elist = dynamic_cast<ExprList *>(initExpr);
        if (elist != NULL) {
            if (elist->exprs.size() == 1)
                lInitSymbol(lvalue, symName, symType, elist->exprs[0], ctx,
                            pos);
            else
                Error(initExpr->pos, "Expression list initializers can't be used for "
                      "variable \"%s\' with type \"%s\".", symName,
                      symType->GetString().c_str());
        }
        return;
    }

    const ReferenceType *rt = dynamic_cast<const ReferenceType *>(symType);
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
        dynamic_cast<const CollectionType *>(symType);
    if (collectionType != NULL) {
        std::string name;
        if (dynamic_cast<const StructType *>(symType) != NULL)
            name = "struct";
        else if (dynamic_cast<const ArrayType *>(symType) != NULL) 
            name = "array";
        else if (dynamic_cast<const VectorType *>(symType) != NULL) 
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
                llvm::Value *ep;
                if (dynamic_cast<const StructType *>(symType) != NULL)
                    ep = ctx->AddElementOffset(lvalue, i, NULL, "element");
                else
                    ep = ctx->GetElementPtrInst(lvalue, LLVMInt32(0), LLVMInt32(i), 
                                                PointerType::GetUniform(collectionType->GetElementType(i)), 
                                                "gep");

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


static bool
lHasUnsizedArrays(const Type *type) {
    const ArrayType *at = dynamic_cast<const ArrayType *>(type);
    if (at == NULL)
        return false;

    if (at->GetElementCount() == 0)
        return true;
    else
        return lHasUnsizedArrays(at->GetElementType());
}
    

void
DeclStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (!ctx->GetCurrentBasicBlock()) 
        return;

    for (unsigned int i = 0; i < vars.size(); ++i) {
        Symbol *sym = vars[i].sym;
        Assert(sym != NULL);
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
        if (dynamic_cast<const ReferenceType *>(sym->type) && initExpr == NULL) {
            Error(sym->pos,
                  "Must provide initializer for reference-type variable \"%s\".",
                  sym->name.c_str());
            continue;
        }

        LLVM_TYPE_CONST llvm::Type *llvmType = sym->type->LLVMType(g->ctx);
        if (llvmType == NULL) {
            Assert(m->errorCount > 0);
            return;
        }

        if (sym->storageClass == SC_STATIC) {
            // For static variables, we need a compile-time constant value
            // for its initializer; if there's no initializer, we use a
            // zero value.
            llvm::Constant *cinit = NULL;
            if (initExpr != NULL) {
                if (lPossiblyResolveFunctionOverloads(initExpr, sym->type) == false)
                    continue;
                // FIXME: we only need this for function pointers; it was
                // already done for atomic types and enums in
                // DeclStmt::TypeCheck()...
                if (dynamic_cast<ExprList *>(initExpr) == NULL) {
                    initExpr = TypeConvertExpr(initExpr, sym->type,
                                               "initializer");
                    // FIXME: and this is only needed to re-establish
                    // constant-ness so that GetConstant below works for
                    // constant artithmetic expressions...
                    initExpr = ::Optimize(initExpr);
                }

                cinit = initExpr->GetConstant(sym->type);
                if (cinit == NULL)
                    Error(initExpr->pos, "Initializer for static variable "
                          "\"%s\" must be a constant.", sym->name.c_str());
            }
            if (cinit == NULL)
                cinit = llvm::Constant::getNullValue(llvmType);

            // Allocate space for the static variable in global scope, so
            // that it persists across function calls
            sym->storagePtr =
                new llvm::GlobalVariable(*m->module, llvmType, 
                                         sym->type->IsConstType(),
                                         llvm::GlobalValue::InternalLinkage, cinit,
                                         llvm::Twine("static.") +
                                         llvm::Twine(sym->pos.first_line) + 
                                         llvm::Twine(".") + sym->name.c_str());
            // Tell the FunctionEmitContext about the variable 
            ctx->EmitVariableDebugInfo(sym);
        }
        else {
            // For non-static variables, allocate storage on the stack
            sym->storagePtr = ctx->AllocaInst(llvmType, sym->name.c_str());

            // Tell the FunctionEmitContext about the variable; must do
            // this before the initializer stuff.
            ctx->EmitVariableDebugInfo(sym);

            // And then get it initialized...
            sym->parentFunction = ctx->GetFunction();
            lInitSymbol(sym->storagePtr, sym->name.c_str(), sym->type, 
                        initExpr, ctx, sym->pos);
        }
    }
}


Stmt *
DeclStmt::Optimize() {
    for (unsigned int i = 0; i < vars.size(); ++i) {
        Expr *init = vars[i].init;
        if (init != NULL && dynamic_cast<ExprList *>(init) == NULL) {
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
            if (sym->type && sym->type->IsConstType() && 
                Type::Equal(init->GetType(), sym->type))
                sym->constValue = dynamic_cast<ConstExpr *>(init);
        }
    }
    return this;
}


Stmt *
DeclStmt::TypeCheck() {
    bool encounteredError = false;
    for (unsigned int i = 0; i < vars.size(); ++i) {
        if (vars[i].sym == NULL) {
            encounteredError = true;
            continue;
        }

        if (vars[i].init == NULL)
            continue;

        // get the right type for stuff like const float foo = 2; so that
        // the int->float type conversion is in there and we don't return
        // an int as the constValue later...
        const Type *type = vars[i].sym->type;
        if (dynamic_cast<const AtomicType *>(type) != NULL ||
            dynamic_cast<const EnumType *>(type) != NULL) {
            // If it's an expr list with an atomic type, we'll later issue
            // an error.  Need to leave vars[i].init as is in that case so
            // it is in fact caught later, though.
            if (dynamic_cast<ExprList *>(vars[i].init) == NULL) {
                vars[i].init = TypeConvertExpr(vars[i].init, type,
                                               "initializer");
                if (vars[i].init == NULL)
                    encounteredError = true;
            }
        }
    }
    return encounteredError ? NULL : this;
}


void
DeclStmt::Print(int indent) const {
    printf("%*cDecl Stmt:", indent, ' ');
    pos.Print();
    for (unsigned int i = 0; i < vars.size(); ++i) {
        printf("%*cVariable %s (%s)", indent+4, ' ', 
               vars[i].sym->name.c_str(),
               vars[i].sym->type->GetString().c_str());
        if (vars[i].init != NULL) {
            printf(" = ");
            vars[i].init->Print();
        }
        printf("\n");
    }
    printf("\n");
}


int
DeclStmt::EstimateCost() const {
    return 0;
}


///////////////////////////////////////////////////////////////////////////
// IfStmt

IfStmt::IfStmt(Expr *t, Stmt *ts, Stmt *fs, bool checkCoherence, SourcePos p) 
    : Stmt(p), test(t), trueStmts(ts), falseStmts(fs), 
      doAllCheck(checkCoherence &&
                 !g->opt.disableCoherentControlFlow) {
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

    llvm::Value *testValue = test->GetValue(ctx);
    if (testValue == NULL)
        return;

    if (isUniform) {
        ctx->StartUniformIf();
        if (doAllCheck)
            Warning(test->pos, "Uniform condition supplied to \"cif\" statement.");

        // 'If' statements with uniform conditions are relatively
        // straightforward.  We evaluate the condition and then jump to
        // either the 'then' or 'else' clause depending on its value.
        llvm::BasicBlock *bthen = ctx->CreateBasicBlock("if_then");
        llvm::BasicBlock *belse = ctx->CreateBasicBlock("if_else");
        llvm::BasicBlock *bexit = ctx->CreateBasicBlock("if_exit");

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
    else
        emitVaryingIf(ctx, testValue);
}


Stmt *
IfStmt::TypeCheck() {
    if (test != NULL) {
        const Type *testType = test->GetType();
        if (testType != NULL) {
            bool isUniform = (testType->IsUniformType() && 
                              !g->opt.disableUniformControlFlow);
            test = TypeConvertExpr(test, isUniform ? AtomicType::UniformBool : 
                                                     AtomicType::VaryingBool, 
                                   "\"if\" statement test");
            if (test == NULL)
                return NULL;
        }
    }

    return this;
}


int
IfStmt::EstimateCost() const {
    const Type *type;
    if (test == NULL || (type = test->GetType()) != NULL)
        return 0;

    return type->IsUniformType() ? COST_UNIFORM_IF : COST_VARYING_IF;
}


void
IfStmt::Print(int indent) const {
    printf("%*cIf Stmt %s", indent, ' ', doAllCheck ? "DO ALL CHECK" : "");
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
    with the mask set appropriately before running each one. 
*/
void
IfStmt::emitMaskedTrueAndFalse(FunctionEmitContext *ctx, llvm::Value *oldMask, 
                               llvm::Value *test) const {
    if (trueStmts) {
        ctx->SetInternalMaskAnd(oldMask, test);
        lEmitIfStatements(ctx, trueStmts, "if: expr mixed, true statements");
        // under varying control flow,, returns can't stop instruction
        // emission, so this better be non-NULL...
        Assert(ctx->GetCurrentBasicBlock()); 
    }
    if (falseStmts) {
        ctx->SetInternalMaskAndNot(oldMask, test);
        lEmitIfStatements(ctx, falseStmts, "if: expr mixed, false statements");
        Assert(ctx->GetCurrentBasicBlock());
    }
}


/** Given an AST node, check to see if it's safe if we happen to run the
    code for that node with the execution mask all off.
 */
static bool
lCheckAllOffSafety(ASTNode *node, void *data) {
    bool *okPtr = (bool *)data;

    if (dynamic_cast<FunctionCallExpr *>(node) != NULL) {
        // FIXME: If we could somehow determine that the function being
        // called was safe (and all of the args Exprs were safe, then it'd
        // be nice to be able to return true here.  (Consider a call to
        // e.g. floatbits() in the stdlib.)  Unfortunately for now we just
        // have to be conservative.
        *okPtr = false;
        return false;
    }

    if (dynamic_cast<AssertStmt *>(node) != NULL) {
        // While it's fine to run the assert for varying tests, it's not
        // desirable to check an assert on a uniform variable if all of the
        // lanes are off.
        *okPtr = false;
        return false;
    }

    if (g->target.allOffMaskIsSafe == true)
        // Don't worry about memory accesses if we have a target that can
        // safely run them with the mask all off
        return true;

    IndexExpr *ie;
    if ((ie = dynamic_cast<IndexExpr *>(node)) != NULL && ie->baseExpr != NULL) {
        const Type *type = ie->baseExpr->GetType();
        if (type == NULL)
            return true;
        if (dynamic_cast<const ReferenceType *>(type) != NULL)
            type = type->GetReferenceTarget();

        ConstExpr *ce = dynamic_cast<ConstExpr *>(ie->index);
        if (ce == NULL) {
            // indexing with a variable... -> not safe
            *okPtr = false;
            return false;
        }

        const PointerType *pointerType = 
            dynamic_cast<const PointerType *>(type);
        if (pointerType != NULL) {
            // pointer[index] -> can't be sure -> not safe
            *okPtr = false;
            return false;
        }

        const SequentialType *seqType = 
            dynamic_cast<const SequentialType *>(type);
        Assert(seqType != NULL);
        int nElements = seqType->GetElementCount();
        if (nElements == 0) {
            // Unsized array, so we can't be sure -> not safe
            *okPtr = false;
            return false;
        }

        int32_t indices[ISPC_MAX_NVEC];
        int count = ce->AsInt32(indices);
        for (int i = 0; i < count; ++i) {
            if (indices[i] < 0 || indices[i] >= nElements) {
                // Index is out of bounds -> not safe
                *okPtr = false;
                return false;
            }
        }

        // All indices are in-bounds
    }

    return true;
}


/** Emit code for an if test that checks the mask and the test values and
    tries to be smart about jumping over code that doesn't need to be run.
 */
void
IfStmt::emitVaryingIf(FunctionEmitContext *ctx, llvm::Value *ltest) const {
    llvm::Value *oldMask = ctx->GetInternalMask();
    if (ctx->GetFullMask() == LLVMMaskAllOn && 
        !g->opt.disableCoherentControlFlow &&
        !g->opt.disableMaskAllOnOptimizations) {
        // We can tell that the mask is on statically at compile time; just
        // emit code for the 'if test with the mask all on' path
        llvm::BasicBlock *bDone = ctx->CreateBasicBlock("cif_done");
        emitMaskAllOn(ctx, ltest, bDone);
        ctx->SetCurrentBasicBlock(bDone);
    }
    else if (doAllCheck) {
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
    }
    else if (trueStmts != NULL || falseStmts != NULL) {
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
        // The lSafeToRunWithAllLanesOff() checks to make sure that we don't do this
        // for potentially dangerous code like:
        //
        // if (index < count) array[index] = 0;
        //
        // where our use of blend for conditional assignments doesn't check
        // for the 'all lanes' off case.
        int trueFalseCost = (::EstimateCost(trueStmts) + 
                             ::EstimateCost(falseStmts));
        bool costIsAcceptable = (trueFalseCost <
                                 PREDICATE_SAFE_IF_STATEMENT_COST);

        bool safeToRunWithAllLanesOff = true;
        WalkAST(trueStmts, lCheckAllOffSafety, NULL, &safeToRunWithAllLanesOff);
        WalkAST(falseStmts, lCheckAllOffSafety, NULL, &safeToRunWithAllLanesOff);

        if (safeToRunWithAllLanesOff &&
            (costIsAcceptable || g->opt.disableCoherentControlFlow)) {
            ctx->StartVaryingIf(oldMask);
            emitMaskedTrueAndFalse(ctx, oldMask, ltest);
            Assert(ctx->GetCurrentBasicBlock());
            ctx->EndIf();
        }
        else {
            llvm::BasicBlock *bDone = ctx->CreateBasicBlock("if_done");
            emitMaskMixed(ctx, oldMask, ltest, bDone);
            ctx->SetCurrentBasicBlock(bDone);
        }
    }
}


/** Emits code for 'if' tests under the case where we know that the program
    mask is all on going into the 'if'.
 */
void
IfStmt::emitMaskAllOn(FunctionEmitContext *ctx, llvm::Value *ltest, 
                      llvm::BasicBlock *bDone) const {
    // We start by explicitly storing "all on" into the mask mask.  Note
    // that this doesn't change its actual value, but doing so lets the
    // compiler see what's going on so that subsequent optimizations for
    // code emitted here can operate with the knowledge that the mask is
    // definitely all on (until it modifies the mask itself).
    Assert(!g->opt.disableCoherentControlFlow);
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
    Assert(ctx->GetCurrentBasicBlock());
    ctx->EndIf();
    ctx->BranchInst(bDone);

    ctx->SetCurrentBasicBlock(bDone);
    ctx->SetFunctionMask(oldFunctionMask);
}


/** Emit code for an 'if' test where the lane mask is known to be mixed
    on/off going into it.
 */
void
IfStmt::emitMaskMixed(FunctionEmitContext *ctx, llvm::Value *oldMask, 
                      llvm::Value *ltest, llvm::BasicBlock *bDone) const {
    ctx->StartVaryingIf(oldMask);
    llvm::BasicBlock *bNext = ctx->CreateBasicBlock("safe_if_after_true");
    if (trueStmts != NULL) {
        llvm::BasicBlock *bRunTrue = ctx->CreateBasicBlock("safe_if_run_true");
        ctx->SetInternalMaskAnd(oldMask, ltest);

        // Do any of the program instances want to run the 'true'
        // block?  If not, jump ahead to bNext.
        llvm::Value *maskAnyQ = ctx->Any(ctx->GetFullMask());
        ctx->BranchInst(bRunTrue, bNext, maskAnyQ);

        // Emit statements for true
        ctx->SetCurrentBasicBlock(bRunTrue);
        lEmitIfStatements(ctx, trueStmts, "if: expr mixed, true statements");
        Assert(ctx->GetCurrentBasicBlock()); 
        ctx->BranchInst(bNext);
        ctx->SetCurrentBasicBlock(bNext);
    }
    if (falseStmts != NULL) {
        llvm::BasicBlock *bRunFalse = ctx->CreateBasicBlock("safe_if_run_false");
        bNext = ctx->CreateBasicBlock("safe_if_after_false");
        ctx->SetInternalMaskAndNot(oldMask, ltest);

        // Similarly, check to see if any of the instances want to
        // run the 'false' block...
        llvm::Value *maskAnyQ = ctx->Any(ctx->GetFullMask());
        ctx->BranchInst(bRunFalse, bNext, maskAnyQ);

        // Emit code for false
        ctx->SetCurrentBasicBlock(bRunFalse);
        lEmitIfStatements(ctx, falseStmts, "if: expr mixed, false statements");
        Assert(ctx->GetCurrentBasicBlock());
        ctx->BranchInst(bNext);
        ctx->SetCurrentBasicBlock(bNext);
    }
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
static bool
lIsVaryingFor(ASTNode *node) {
    IfStmt *ifStmt;
    if ((ifStmt = dynamic_cast<IfStmt *>(node)) != NULL &&
        ifStmt->test != NULL) {
        const Type *type = ifStmt->test->GetType();
        return (type != NULL && type->IsVaryingType());
    }
    else
        return false;
}


/** Preorder callback function for checking for varying breaks or
    continues. */
static bool
lVaryingBCPreFunc(ASTNode *node, void *d) {
    VaryingBCCheckInfo *info = (VaryingBCCheckInfo *)d;

    // We found a break or continue statement; if we're under varying
    // control flow, then bingo.
    if ((dynamic_cast<BreakStmt *>(node) != NULL ||
         dynamic_cast<ContinueStmt *>(node) != NULL) &&
        info->varyingControlFlowDepth > 0) {
        info->foundVaryingBreakOrContinue = true;
        return false;
    }

    // Update the count of the nesting depth of varying control flow if
    // this is an if statement with a varying condition.
    if (lIsVaryingFor(node))
        ++info->varyingControlFlowDepth;

    if (dynamic_cast<ForStmt *>(node) != NULL ||
        dynamic_cast<DoStmt *>(node) != NULL ||
        dynamic_cast<ForeachStmt *>(node) != NULL)
        // Don't recurse into these guys, since we don't care about varying
        // breaks or continues within them...
        return false;
    else
        return true;
}


/** Postorder callback function for checking for varying breaks or
    continues; decrement the varying control flow depth after the node's
    children have been processed, if this is a varying if statement. */
static ASTNode *
lVaryingBCPostFunc(ASTNode *node, void *d) {
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
static bool
lHasVaryingBreakOrContinue(Stmt *stmt) {
    VaryingBCCheckInfo info;
    WalkAST(stmt, lVaryingBCPreFunc, lVaryingBCPostFunc, &info);
    return info.foundVaryingBreakOrContinue;
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

    ctx->StartLoop(bexit, btest, uniformTest);

    // Start by jumping into the loop body
    ctx->BranchInst(bloop);

    // And now emit code for the loop body
    ctx->SetCurrentBasicBlock(bloop);
    ctx->SetLoopMask(ctx->GetInternalMask());
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
        if (!g->opt.disableMaskAllOnOptimizations)
            ctx->SetInternalMask(LLVMMaskAllOn);
        llvm::Value *oldFunctionMask = ctx->GetFunctionMask();
        if (!g->opt.disableMaskAllOnOptimizations)
            ctx->SetFunctionMask(LLVMMaskAllOn);
        if (bodyStmts)
            bodyStmts->EmitCode(ctx);
        Assert(ctx->GetCurrentBasicBlock());
        ctx->SetFunctionMask(oldFunctionMask);
        ctx->BranchInst(btest);

        // The mask is mixed.  Just emit the code for the loop body.
        ctx->SetCurrentBasicBlock(bMixed);
        if (bodyStmts)
            bodyStmts->EmitCode(ctx);
        Assert(ctx->GetCurrentBasicBlock());
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
        llvm::Value *mask = ctx->GetInternalMask();
        ctx->SetInternalMaskAnd(mask, testValue);
        ctx->BranchIfMaskAny(bloop, bexit);
    }

    // ...and we're done.  Set things up for subsequent code to be emitted
    // in the right basic block.
    ctx->SetCurrentBasicBlock(bexit);
    ctx->EndLoop();
}


Stmt *
DoStmt::TypeCheck() {
    const Type *testType;
    if (testExpr != NULL && (testType = testExpr->GetType()) != NULL) {
        if (!testType->IsNumericType() && !testType->IsBoolType()) {
            Error(testExpr->pos, "Type \"%s\" can't be converted to boolean for \"while\" "
                  "test in \"do\" loop.", testExpr->GetType()->GetString().c_str());
            return NULL;
        }

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
        bool uniformTest = (testType->IsUniformType() &&
                            !g->opt.disableUniformControlFlow &&
                            !lHasVaryingBreakOrContinue(bodyStmts));
        testExpr = new TypeCastExpr(uniformTest ? AtomicType::UniformBool :
                                                  AtomicType::VaryingBool,
                                    testExpr, testExpr->pos);
    }

    return this;
}


int
DoStmt::EstimateCost() const {
    bool uniformTest = testExpr ? testExpr->GetType()->IsUniformType() :
        (!g->opt.disableUniformControlFlow &&
         !lHasVaryingBreakOrContinue(bodyStmts));

    return uniformTest ? COST_UNIFORM_LOOP : COST_VARYING_LOOP;
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

    ctx->StartLoop(bexit, bstep, uniformTest);
    ctx->SetDebugPos(pos);

    // If we have an initiailizer statement, start by emitting the code for
    // it and then jump into the loop test code.  (Also start a new scope
    // since the initiailizer may be a declaration statement).
    if (init) {
        Assert(dynamic_cast<StmtList *>(init) == NULL);
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
        Assert(ltest->getType() == LLVMTypes::BoolType);
        ctx->BranchInst(bloop, bexit, ltest);
    }
    else {
        llvm::Value *mask = ctx->GetInternalMask();
        ctx->SetInternalMaskAnd(mask, ltest);
        ctx->BranchIfMaskAny(bloop, bexit);
    }

    // On to emitting the code for the loop body.
    ctx->SetCurrentBasicBlock(bloop);
    ctx->SetLoopMask(ctx->GetInternalMask());
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
        if (!g->opt.disableMaskAllOnOptimizations)
            ctx->SetInternalMask(LLVMMaskAllOn);
        llvm::Value *oldFunctionMask = ctx->GetFunctionMask();
        if (!g->opt.disableMaskAllOnOptimizations)
            ctx->SetFunctionMask(LLVMMaskAllOn);
        if (stmts)
            stmts->EmitCode(ctx);
        Assert(ctx->GetCurrentBasicBlock());
        ctx->SetFunctionMask(oldFunctionMask);
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
ForStmt::TypeCheck() {
    const Type *testType;
    if (test && (testType = test->GetType()) != NULL) {
        if (!testType->IsNumericType() && !testType->IsBoolType()) {
            Error(test->pos, "Type \"%s\" can't be converted to boolean for \"for\" "
                  "loop test.", test->GetType()->GetString().c_str());
            return NULL;
        }

        // See comments in DoStmt::TypeCheck() regarding
        // 'uniformTest' and the type cast here.
        bool uniformTest = (testType->IsUniformType() &&
                            !g->opt.disableUniformControlFlow &&
                            !lHasVaryingBreakOrContinue(stmts));
        test = new TypeCastExpr(uniformTest ? AtomicType::UniformBool :
                                AtomicType::VaryingBool, test, test->pos);
        test = ::TypeCheck(test);
        if (test == NULL)
            return NULL;
    }

    return this;
}


int
ForStmt::EstimateCost() const {
    bool uniformTest = test ? test->GetType()->IsUniformType() :
        (!g->opt.disableUniformControlFlow &&
         !lHasVaryingBreakOrContinue(stmts));

    return uniformTest ? COST_UNIFORM_LOOP : COST_VARYING_LOOP;
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
BreakStmt::TypeCheck() {
    return this;
}


int
BreakStmt::EstimateCost() const {
    return doCoherenceCheck ? COST_COHERENT_BREAK_CONTINE : 
        COST_REGULAR_BREAK_CONTINUE;
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
ContinueStmt::TypeCheck() {
    return this;
}


int
ContinueStmt::EstimateCost() const {
    return doCoherenceCheck ? COST_COHERENT_BREAK_CONTINE : 
        COST_REGULAR_BREAK_CONTINUE;
}


void
ContinueStmt::Print(int indent) const {
    printf("%*c%sContinue Stmt", indent, ' ', doCoherenceCheck ? "Coherent " : "");
    pos.Print();
    printf("\n");
}


///////////////////////////////////////////////////////////////////////////
// ForeachStmt

ForeachStmt::ForeachStmt(const std::vector<Symbol *> &lvs, 
                         const std::vector<Expr *> &se, 
                         const std::vector<Expr *> &ee, 
                         Stmt *s, bool t, SourcePos pos)
    : Stmt(pos), dimVariables(lvs), startExprs(se), endExprs(ee), isTiled(t),
      stmts(s) {
}


/* Given a uniform counter value in the memory location pointed to by
   uniformCounterPtr, compute the corresponding set of varying counter
   values for use within the loop body.
 */
static llvm::Value *
lUpdateVaryingCounter(int dim, int nDims, FunctionEmitContext *ctx, 
                      llvm::Value *uniformCounterPtr,
                      llvm::Value *varyingCounterPtr,
                      const std::vector<int> &spans) {
    // Smear the uniform counter value out to be varying
    llvm::Value *counter = ctx->LoadInst(uniformCounterPtr);
    llvm::Value *smearCounter = 
        llvm::UndefValue::get(LLVMTypes::Int32VectorType);
    for (int i = 0; i < g->target.vectorWidth; ++i)
        smearCounter = 
            ctx->InsertInst(smearCounter, counter, i, "smear_counter");

    // Figure out the offsets; this is a little bit tricky.  As an example,
    // consider a 2D tiled foreach loop, where we're running 8-wide and
    // where the inner dimension has a stride of 4 and the outer dimension
    // has a stride of 2.  For the inner dimension, we want the offsets
    // (0,1,2,3,0,1,2,3), and for the outer dimension we want
    // (0,0,0,0,1,1,1,1).
    int32_t delta[ISPC_MAX_NVEC];
    for (int i = 0; i < g->target.vectorWidth; ++i) {
        int d = i;
        // First, account for the effect of any dimensions at deeper
        // nesting levels than the current one.
        int prevDimSpanCount = 1;
        for (int j = dim; j < nDims-1; ++j)
            prevDimSpanCount *= spans[j+1];
        d /= prevDimSpanCount;

        // And now with what's left, figure out our own offset
        delta[i] = d % spans[dim];
    }

    // Add the deltas to compute the varying counter values; store the
    // result to memory and then return it directly as well.
    llvm::Value *varyingCounter = 
        ctx->BinaryOperator(llvm::Instruction::Add, smearCounter,
                            LLVMInt32Vector(delta), "iter_val");
    ctx->StoreInst(varyingCounter, varyingCounterPtr);
    return varyingCounter;
}


/** Returns the integer log2 of the given integer. */
static int
lLog2(int i) {
    int ret = 0;
    while (i != 0) {
        ++ret;
        i >>= 1;
    }
    return ret-1;
}


/* Figure out how many elements to process in each dimension for each time
   through a foreach loop.  The untiled case is easy; all of the outer
   dimensions up until the innermost one have a span of 1, and the
   innermost one takes the entire vector width.  For the tiled case, we
   give wider spans to the innermost dimensions while also trying to
   generate relatively square domains.

   This code works recursively from outer dimensions to inner dimensions.
 */
static void
lGetSpans(int dimsLeft, int nDims, int itemsLeft, bool isTiled, int *a) {
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

    lGetSpans(dimsLeft-1, nDims, itemsLeft / *a, isTiled, a+1);
}


/* Emit code for a foreach statement.  We effectively emit code to run the
   set of n-dimensional nested loops corresponding to the dimensionality of
   the foreach statement along with the extra logic to deal with mismatches
   between the vector width we're compiling to and the number of elements
   to process.
 */
void
ForeachStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (ctx->GetCurrentBasicBlock() == NULL || stmts == NULL) 
        return;

    llvm::BasicBlock *bbCheckExtras = ctx->CreateBasicBlock("foreach_check_extras");
    llvm::BasicBlock *bbDoExtras = ctx->CreateBasicBlock("foreach_do_extras");
    llvm::BasicBlock *bbBody = ctx->CreateBasicBlock("foreach_body");
    llvm::BasicBlock *bbExit = ctx->CreateBasicBlock("foreach_exit");

    llvm::Value *oldMask = ctx->GetInternalMask();

    ctx->SetDebugPos(pos);
    ctx->StartScope();

    // This should be caught during typechecking
    Assert(startExprs.size() == dimVariables.size() && 
           endExprs.size() == dimVariables.size());
    int nDims = (int)dimVariables.size();

    ///////////////////////////////////////////////////////////////////////
    // Setup: compute the number of items we have to work on in each
    // dimension and a number of derived values.
    std::vector<llvm::BasicBlock *> bbReset, bbStep, bbTest;
    std::vector<llvm::Value *> startVals, endVals, uniformCounterPtrs;
    std::vector<llvm::Value *> nItems, nExtras, alignedEnd;
    std::vector<llvm::Value *> extrasMaskPtrs;

    std::vector<int> span(nDims, 0);
    lGetSpans(nDims-1, nDims, g->target.vectorWidth, isTiled, &span[0]);

    for (int i = 0; i < nDims; ++i) {
        // Basic blocks that we'll fill in later with the looping logic for
        // this dimension.
        bbReset.push_back(ctx->CreateBasicBlock("foreach_reset"));
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
        nItems.push_back(ctx->BinaryOperator(llvm::Instruction::Sub, ev, sv,
                                             "nitems"));

        // nExtras = nItems % (span for this dimension)
        // This gives us the number of extra elements we need to deal with
        // at the end of the loop for this dimension that don't fit cleanly
        // into a vector width.
        nExtras.push_back(ctx->BinaryOperator(llvm::Instruction::SRem, nItems[i],
                                              LLVMInt32(span[i]), "nextras"));

        // alignedEnd = endVal - nExtras
        alignedEnd.push_back(ctx->BinaryOperator(llvm::Instruction::Sub, ev,
                                                 nExtras[i], "aligned_end"));

        ///////////////////////////////////////////////////////////////////////
        // Each dimension has a loop counter that is a uniform value that
        // goes from startVal to endVal, in steps of the span for this
        // dimension.  Its value is only used internally here for looping
        // logic and isn't directly available in the user's program code.
        uniformCounterPtrs.push_back(ctx->AllocaInst(LLVMTypes::Int32Type, 
                                                     "counter"));
        ctx->StoreInst(startVals[i], uniformCounterPtrs[i]);

        // There is also a varying variable that holds the set of index
        // values for each dimension in the current loop iteration; this is
        // the value that is program-visible.
        dimVariables[i]->storagePtr = ctx->AllocaInst(LLVMTypes::Int32VectorType, 
                                                  dimVariables[i]->name.c_str());
        dimVariables[i]->parentFunction = ctx->GetFunction();
        ctx->EmitVariableDebugInfo(dimVariables[i]);

        // Each dimension also maintains a mask that represents which of
        // the varying elements in the current iteration should be
        // processed.  (i.e. this is used to disable the lanes that have
        // out-of-bounds offsets.)
        extrasMaskPtrs.push_back(ctx->AllocaInst(LLVMTypes::MaskType, "extras mask"));
        ctx->StoreInst(LLVMMaskAllOn, extrasMaskPtrs[i]);
    }

    ctx->StartForeach(bbStep[nDims-1]);

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
            ctx->BranchInst(bbStep[i-1]);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // foreach_test
    std::vector<llvm::Value *> inExtras;
    for (int i = 0; i < nDims; ++i) {
        ctx->SetCurrentBasicBlock(bbTest[i]);

        llvm::Value *haveExtras = 
            ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SGT,
                         endVals[i], alignedEnd[i], "have_extras");

        llvm::Value *counter = ctx->LoadInst(uniformCounterPtrs[i], "counter");
        llvm::Value *atAlignedEnd = 
            ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ,
                         counter, alignedEnd[i], "at_aligned_end");
        llvm::Value *inEx = 
            ctx->BinaryOperator(llvm::Instruction::And, haveExtras,
                                atAlignedEnd, "in_extras");

        if (i == 0)
            inExtras.push_back(inEx);
        else
            inExtras.push_back(ctx->BinaryOperator(llvm::Instruction::Or, inEx,
                                                   inExtras[i-1], "in_extras_all"));

        llvm::Value *varyingCounter = 
            lUpdateVaryingCounter(i, nDims, ctx, uniformCounterPtrs[i], 
                                  dimVariables[i]->storagePtr, span);

        llvm::Value *smearEnd = llvm::UndefValue::get(LLVMTypes::Int32VectorType);
        for (int j = 0; j < g->target.vectorWidth; ++j)
            smearEnd = ctx->InsertInst(smearEnd, endVals[i], j, "smear_end");
        // Do a vector compare of its value to the end value to generate a
        // mask for this last bit of work.
        llvm::Value *emask = 
            ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SLT,
                         varyingCounter, smearEnd);
        emask = ctx->I1VecToBoolVec(emask);

        if (i == 0)
            ctx->StoreInst(emask, extrasMaskPtrs[i]);
        else {
            // FIXME: at least specialize the innermost loop to not do all
            // this mask stuff each time through the test...
            llvm::Value *oldMask = ctx->LoadInst(extrasMaskPtrs[i-1]);
            llvm::Value *newMask =
                ctx->BinaryOperator(llvm::Instruction::And, oldMask, emask,
                                    "extras_mask");
            ctx->StoreInst(newMask, extrasMaskPtrs[i]);
        }

        llvm::Value *notAtEnd = 
            ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SLT,
                         counter, endVals[i]);
        if (i != nDims-1)
            ctx->BranchInst(bbTest[i+1], bbReset[i], notAtEnd);
        else
            ctx->BranchInst(bbCheckExtras, bbReset[i], notAtEnd);
    }

    ///////////////////////////////////////////////////////////////////////////
    // foreach_step: increment the uniform counter by the vector width.
    // Note that we don't increment the varying counter here as well but
    // just generate its value when we need it in the loop body.
    for (int i = 0; i < nDims; ++i) {
        ctx->SetCurrentBasicBlock(bbStep[i]);
        if (i == nDims-1)
            ctx->RestoreContinuedLanes();
        llvm::Value *counter = ctx->LoadInst(uniformCounterPtrs[i]);
        llvm::Value *newCounter =  
            ctx->BinaryOperator(llvm::Instruction::Add, counter,
                                LLVMInt32(span[i]), "new_counter");
        ctx->StoreInst(newCounter, uniformCounterPtrs[i]);
        ctx->BranchInst(bbTest[i]);
    }

    ///////////////////////////////////////////////////////////////////////////
    // foreach_check_extras: see if we need to deal with any partial
    // vector's worth of work that's left.
    ctx->SetCurrentBasicBlock(bbCheckExtras);
    ctx->AddInstrumentationPoint("foreach loop check extras");
    ctx->BranchInst(bbDoExtras, bbBody, inExtras[nDims-1]);

    ///////////////////////////////////////////////////////////////////////////
    // foreach_body: do a full vector's worth of work.  We know that all
    // lanes will be running here, so we explicitly set the mask to be 'all
    // on'.  This ends up being relatively straightforward: just update the
    // value of the varying loop counter and have the statements in the
    // loop body emit their code.
    ctx->SetCurrentBasicBlock(bbBody);
    ctx->SetInternalMask(LLVMMaskAllOn);
    ctx->AddInstrumentationPoint("foreach loop body");
    stmts->EmitCode(ctx);
    Assert(ctx->GetCurrentBasicBlock() != NULL);
    ctx->BranchInst(bbStep[nDims-1]);

    ///////////////////////////////////////////////////////////////////////////
    // foreach_doextras: set the mask and have the statements emit their
    // code again.  Note that it's generally worthwhile having two copies
    // of the statements' code, since the code above is emitted with the
    // mask known to be all-on, which in turn leads to more efficient code
    // for that case.
    ctx->SetCurrentBasicBlock(bbDoExtras);
    llvm::Value *mask = ctx->LoadInst(extrasMaskPtrs[nDims-1]);
    ctx->SetInternalMask(mask);
    stmts->EmitCode(ctx);
    ctx->BranchInst(bbStep[nDims-1]);

    ///////////////////////////////////////////////////////////////////////////
    // foreach_exit: All done.  Restore the old mask and clean up
    ctx->SetCurrentBasicBlock(bbExit);
    ctx->SetInternalMask(oldMask);

    ctx->EndForeach();
    ctx->EndScope();
}


Stmt *
ForeachStmt::TypeCheck() {
    bool anyErrors = false;
    for (unsigned int i = 0; i < startExprs.size(); ++i) {
        if (startExprs[i] != NULL)
            startExprs[i] = TypeConvertExpr(startExprs[i], 
                                            AtomicType::UniformInt32, 
                                            "foreach starting value");
        anyErrors |= (startExprs[i] == NULL);
    }
    for (unsigned int i = 0; i < endExprs.size(); ++i) {
        if (endExprs[i] != NULL)
            endExprs[i] = TypeConvertExpr(endExprs[i], AtomicType::UniformInt32,
                                          "foreach ending value");
        anyErrors |= (endExprs[i] == NULL);
    }

    if (startExprs.size() < dimVariables.size()) {
        Error(pos, "Not enough initial values provided for \"foreach\" loop; "
              "got %d, expected %d\n", (int)startExprs.size(), (int)dimVariables.size());
        anyErrors = true;
    }
    else if (startExprs.size() > dimVariables.size()) {
        Error(pos, "Too many initial values provided for \"foreach\" loop; "
              "got %d, expected %d\n", (int)startExprs.size(), (int)dimVariables.size());
        anyErrors = true;
    }

    if (endExprs.size() < dimVariables.size()) {
        Error(pos, "Not enough initial values provided for \"foreach\" loop; "
              "got %d, expected %d\n", (int)endExprs.size(), (int)dimVariables.size());
        anyErrors = true;
    }
    else if (endExprs.size() > dimVariables.size()) {
        Error(pos, "Too many initial values provided for \"foreach\" loop; "
              "got %d, expected %d\n", (int)endExprs.size(), (int)dimVariables.size());
        anyErrors = true;
    }

    return anyErrors ? NULL : this;
}


int
ForeachStmt::EstimateCost() const {
    return dimVariables.size() * (COST_UNIFORM_LOOP + COST_SIMPLE_ARITH_LOGIC_OP);
}


void
ForeachStmt::Print(int indent) const {
    printf("%*cForeach Stmt", indent, ' ');
    pos.Print();
    printf("\n");
    
    for (unsigned int i = 0; i < dimVariables.size(); ++i)
        if (dimVariables[i] != NULL)
            printf("%*cVar %d: %s\n", indent+4, ' ', i, 
                   dimVariables[i]->name.c_str());
        else 
            printf("%*cVar %d: NULL\n", indent+4, ' ', i);

    printf("Start values:\n");
    for (unsigned int i = 0; i < startExprs.size(); ++i) {
        if (startExprs[i] != NULL)
            startExprs[i]->Print();
        else
            printf("NULL");
        if (i != startExprs.size()-1)
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
        if (i != endExprs.size()-1)
            printf(", ");
        else
            printf("\n");
    }

    if (stmts != NULL) {
        printf("%*cStmts:\n", indent+4, ' ');
        stmts->Print(indent+8);
    }
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

    if (ctx->InForeachLoop()) {
        Error(pos, "\"return\" statement is illegal inside a \"foreach\" loop.");
        return;
    }

    ctx->SetDebugPos(pos);
    ctx->CurrentLanesReturned(val, doCoherenceCheck);
}


Stmt *
ReturnStmt::TypeCheck() {
    return this;
}


int
ReturnStmt::EstimateCost() const {
    return COST_RETURN;
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
// GotoStmt

GotoStmt::GotoStmt(const char *l, SourcePos gotoPos, SourcePos ip) 
    : Stmt(gotoPos) {
    label = l;
    identifierPos = ip;
}


void
GotoStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (ctx->VaryingCFDepth() > 0) {
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
        // TODO: use the string distance stuff to suggest alternatives if
        // there are some with names close to the label name we have here..
        Error(identifierPos, "No label named \"%s\" found in current function.",
              label.c_str());
        return;
    }

    ctx->BranchInst(bb);
    ctx->SetCurrentBasicBlock(NULL);
}


void
GotoStmt::Print(int indent) const {
    printf("%*cGoto label \"%s\"\n", indent, ' ', label.c_str());
}


Stmt *
GotoStmt::Optimize() {
    return this;
}


Stmt *
GotoStmt::TypeCheck() {
    return this;
}


int
GotoStmt::EstimateCost() const {
    return COST_GOTO;
}


///////////////////////////////////////////////////////////////////////////
// LabeledStmt

LabeledStmt::LabeledStmt(const char *n, Stmt *s, SourcePos p) 
    : Stmt(p) {
    name = n;
    stmt = s;
}


void
LabeledStmt::EmitCode(FunctionEmitContext *ctx) const {
    llvm::BasicBlock *bblock = ctx->GetLabeledBasicBlock(name);
    Assert(bblock != NULL);

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


void
LabeledStmt::Print(int indent) const {
    printf("%*cLabel \"%s\"\n", indent, ' ', name.c_str());
    if (stmt != NULL)
        stmt->Print(indent);
}


Stmt *
LabeledStmt::Optimize() {
    return this;
}


Stmt *
LabeledStmt::TypeCheck() {
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


int
LabeledStmt::EstimateCost() const {
    return 0;
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
StmtList::TypeCheck() {
    return this;
}


int
StmtList::EstimateCost() const {
    return 0;
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
    if (dynamic_cast<const PointerType *>(t) != NULL) {
        if (t->IsUniformType())
            return 'p';
        else
            return 'P';
    }
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

    // Just int8 and int16 types to int32s...
    const Type *baseType = type->GetAsNonConstType()->GetAsUniformType();
    if (baseType == AtomicType::UniformInt8 ||
        baseType == AtomicType::UniformUInt8 ||
        baseType == AtomicType::UniformInt16 ||
        baseType == AtomicType::UniformUInt16) {
        expr = new TypeCastExpr(type->IsUniformType() ? AtomicType::UniformInt32 :
                                                        AtomicType::VaryingInt32, 
                                expr, expr->pos);
        type = expr->GetType();
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
   the builtins-c.c file.  In particular, the EmitCode() method here needs to
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

    if (values == NULL) {
        LLVM_TYPE_CONST llvm::Type *ptrPtrType = 
            llvm::PointerType::get(LLVMTypes::VoidPointerType, 0);
        args[4] = llvm::Constant::getNullValue(ptrPtrType);
    }
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

                llvm::Value *arrayPtr = ctx->AddElementOffset(argPtrArray, i, NULL);
                ctx->StoreInst(ptr, arrayPtr);
            }
        }
        else {
            llvm::Value *ptr = lProcessPrintArg(values, ctx, argTypes);
            if (!ptr)
                return;
            llvm::Value *arrayPtr = ctx->AddElementOffset(argPtrArray, 0, NULL);
            ctx->StoreInst(ptr, arrayPtr);
        }
    }

    // Now we can emit code to call __do_print()
    llvm::Function *printFunc = m->module->getFunction("__do_print");
    Assert(printFunc);

    llvm::Value *mask = ctx->GetFullMask();
    // Set up the rest of the parameters to it
    args[0] = ctx->GetStringPtr(format);
    args[1] = ctx->GetStringPtr(argTypes);
    args[2] = LLVMInt32(g->target.vectorWidth);
    args[3] = ctx->LaneMask(mask);
    std::vector<llvm::Value *> argVec(&args[0], &args[5]);
    ctx->CallInst(printFunc, NULL, argVec, "");
}


void
PrintStmt::Print(int indent) const {
    printf("%*cPrint Stmt (%s)", indent, ' ', format.c_str());
}


Stmt *
PrintStmt::TypeCheck() {
    return this;
}


int
PrintStmt::EstimateCost() const {
    return COST_FUNCALL;
}


///////////////////////////////////////////////////////////////////////////
// AssertStmt

AssertStmt::AssertStmt(const std::string &msg, Expr *e, SourcePos p) 
    : Stmt(p), message(msg), expr(e) {
}


void
AssertStmt::EmitCode(FunctionEmitContext *ctx) const {
    if (expr == NULL)
        return;
    const Type *type = expr->GetType();
    if (type == NULL)
        return;
    bool isUniform = type->IsUniformType();

    // The actual functionality to do the check and then handle falure is
    // done via a builtin written in bitcode in builtins.m4.
    llvm::Function *assertFunc = 
        isUniform ? m->module->getFunction("__do_assert_uniform") :
                    m->module->getFunction("__do_assert_varying");
    Assert(assertFunc != NULL);

    char *errorString;
    if (asprintf(&errorString, "%s:%d:%d: Assertion failed: %s\n", 
                 pos.name, pos.first_line, pos.first_column, 
                 message.c_str()) == -1) {
        Error(pos, "Fatal error when generating assert string: asprintf() "
              "unable to allocate memory!");
        return;
    }

    std::vector<llvm::Value *> args;
    args.push_back(ctx->GetStringPtr(errorString));
    args.push_back(expr->GetValue(ctx));
    args.push_back(ctx->GetFullMask());
    ctx->CallInst(assertFunc, NULL, args, "");

    free(errorString);
}


void
AssertStmt::Print(int indent) const {
    printf("%*cAssert Stmt (%s)", indent, ' ', message.c_str());
}


Stmt *
AssertStmt::TypeCheck() {
    const Type *type;
    if (expr && (type = expr->GetType()) != NULL) {
        bool isUniform = type->IsUniformType();
        if (!type->IsNumericType() && !type->IsBoolType()) {
            Error(expr->pos, "Type \"%s\" can't be converted to boolean for \"assert\".",
                  type->GetString().c_str());
            return NULL;
        }
        expr = new TypeCastExpr(isUniform ? AtomicType::UniformBool : 
                                            AtomicType::VaryingBool, 
                                expr, expr->pos);
        expr = ::TypeCheck(expr);
    }
    return this;
}


int
AssertStmt::EstimateCost() const {
    return COST_ASSERT;
}

