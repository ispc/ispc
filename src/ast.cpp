/*
  Copyright (c) 2011-2021, Intel Corporation
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

/** @file ast.cpp

    @brief General functionality related to abstract syntax trees and
    traversal of them.
 */

#include "ast.h"
#include "expr.h"
#include "func.h"
#include "stmt.h"
#include "sym.h"
#include "util.h"

#include <llvm/Support/TimeProfiler.h>

using namespace ispc;

///////////////////////////////////////////////////////////////////////////
// ASTNode

ASTNode::~ASTNode() {}

///////////////////////////////////////////////////////////////////////////
// AST

void AST::AddFunction(Symbol *sym, Stmt *code) {
    if (sym == NULL)
        return;
    functions.push_back(new Function(sym, code));
}

void AST::GenerateIR() {
    llvm::TimeTraceScope TimeScope("GenerateIR");
    for (unsigned int i = 0; i < functions.size(); ++i)
        functions[i]->GenerateIR();
}

///////////////////////////////////////////////////////////////////////////

ASTNode *ispc::WalkAST(ASTNode *node, ASTPreCallBackFunc preFunc, ASTPostCallBackFunc postFunc, void *data) {
    if (node == NULL)
        return node;

    // Call the callback function
    if (preFunc != NULL) {
        if (preFunc(node, data) == false)
            // The function asked us to not continue recursively, so stop.
            return node;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Handle Statements
    if (llvm::dyn_cast<Stmt>(node) != NULL) {
        ExprStmt *es;
        DeclStmt *ds;
        IfStmt *is;
        DoStmt *dos;
        ForStmt *fs;
        ForeachStmt *fes;
        ForeachActiveStmt *fas;
        ForeachUniqueStmt *fus;
        CaseStmt *cs;
        DefaultStmt *defs;
        SwitchStmt *ss;
        ReturnStmt *rs;
        LabeledStmt *ls;
        StmtList *sl;
        PrintStmt *ps;
        AssertStmt *as;
        DeleteStmt *dels;
        UnmaskedStmt *ums;

        if ((es = llvm::dyn_cast<ExprStmt>(node)) != NULL)
            es->expr = (Expr *)WalkAST(es->expr, preFunc, postFunc, data);
        else if ((ds = llvm::dyn_cast<DeclStmt>(node)) != NULL) {
            for (unsigned int i = 0; i < ds->vars.size(); ++i)
                ds->vars[i].init = (Expr *)WalkAST(ds->vars[i].init, preFunc, postFunc, data);
        } else if ((is = llvm::dyn_cast<IfStmt>(node)) != NULL) {
            is->test = (Expr *)WalkAST(is->test, preFunc, postFunc, data);
            is->trueStmts = (Stmt *)WalkAST(is->trueStmts, preFunc, postFunc, data);
            is->falseStmts = (Stmt *)WalkAST(is->falseStmts, preFunc, postFunc, data);
        } else if ((dos = llvm::dyn_cast<DoStmt>(node)) != NULL) {
            dos->testExpr = (Expr *)WalkAST(dos->testExpr, preFunc, postFunc, data);
            dos->bodyStmts = (Stmt *)WalkAST(dos->bodyStmts, preFunc, postFunc, data);
        } else if ((fs = llvm::dyn_cast<ForStmt>(node)) != NULL) {
            fs->init = (Stmt *)WalkAST(fs->init, preFunc, postFunc, data);
            fs->test = (Expr *)WalkAST(fs->test, preFunc, postFunc, data);
            fs->step = (Stmt *)WalkAST(fs->step, preFunc, postFunc, data);
            fs->stmts = (Stmt *)WalkAST(fs->stmts, preFunc, postFunc, data);
        } else if ((fes = llvm::dyn_cast<ForeachStmt>(node)) != NULL) {
            for (unsigned int i = 0; i < fes->startExprs.size(); ++i)
                fes->startExprs[i] = (Expr *)WalkAST(fes->startExprs[i], preFunc, postFunc, data);
            for (unsigned int i = 0; i < fes->endExprs.size(); ++i)
                fes->endExprs[i] = (Expr *)WalkAST(fes->endExprs[i], preFunc, postFunc, data);
            fes->stmts = (Stmt *)WalkAST(fes->stmts, preFunc, postFunc, data);
        } else if ((fas = llvm::dyn_cast<ForeachActiveStmt>(node)) != NULL) {
            fas->stmts = (Stmt *)WalkAST(fas->stmts, preFunc, postFunc, data);
        } else if ((fus = llvm::dyn_cast<ForeachUniqueStmt>(node)) != NULL) {
            fus->expr = (Expr *)WalkAST(fus->expr, preFunc, postFunc, data);
            fus->stmts = (Stmt *)WalkAST(fus->stmts, preFunc, postFunc, data);
        } else if ((cs = llvm::dyn_cast<CaseStmt>(node)) != NULL)
            cs->stmts = (Stmt *)WalkAST(cs->stmts, preFunc, postFunc, data);
        else if ((defs = llvm::dyn_cast<DefaultStmt>(node)) != NULL)
            defs->stmts = (Stmt *)WalkAST(defs->stmts, preFunc, postFunc, data);
        else if ((ss = llvm::dyn_cast<SwitchStmt>(node)) != NULL) {
            ss->expr = (Expr *)WalkAST(ss->expr, preFunc, postFunc, data);
            ss->stmts = (Stmt *)WalkAST(ss->stmts, preFunc, postFunc, data);
        } else if (llvm::dyn_cast<BreakStmt>(node) != NULL || llvm::dyn_cast<ContinueStmt>(node) != NULL ||
                   llvm::dyn_cast<GotoStmt>(node) != NULL) {
            // nothing
        } else if ((ls = llvm::dyn_cast<LabeledStmt>(node)) != NULL)
            ls->stmt = (Stmt *)WalkAST(ls->stmt, preFunc, postFunc, data);
        else if ((rs = llvm::dyn_cast<ReturnStmt>(node)) != NULL)
            rs->expr = (Expr *)WalkAST(rs->expr, preFunc, postFunc, data);
        else if ((sl = llvm::dyn_cast<StmtList>(node)) != NULL) {
            std::vector<Stmt *> &sls = sl->stmts;
            for (unsigned int i = 0; i < sls.size(); ++i)
                sls[i] = (Stmt *)WalkAST(sls[i], preFunc, postFunc, data);
        } else if ((ps = llvm::dyn_cast<PrintStmt>(node)) != NULL)
            ps->values = (Expr *)WalkAST(ps->values, preFunc, postFunc, data);
        else if ((as = llvm::dyn_cast<AssertStmt>(node)) != NULL)
            as->expr = (Expr *)WalkAST(as->expr, preFunc, postFunc, data);
        else if ((dels = llvm::dyn_cast<DeleteStmt>(node)) != NULL)
            dels->expr = (Expr *)WalkAST(dels->expr, preFunc, postFunc, data);
        else if ((ums = llvm::dyn_cast<UnmaskedStmt>(node)) != NULL)
            ums->stmts = (Stmt *)WalkAST(ums->stmts, preFunc, postFunc, data);
        else
            FATAL("Unhandled statement type in WalkAST()");
    } else {
        ///////////////////////////////////////////////////////////////////////////
        // Handle expressions
        Assert(llvm::dyn_cast<Expr>(node) != NULL);
        UnaryExpr *ue;
        BinaryExpr *be;
        AssignExpr *ae;
        SelectExpr *se;
        ExprList *el;
        FunctionCallExpr *fce;
        IndexExpr *ie;
        MemberExpr *me;
        TypeCastExpr *tce;
        ReferenceExpr *re;
        PtrDerefExpr *ptrderef;
        RefDerefExpr *refderef;
        SizeOfExpr *soe;
        AddressOfExpr *aoe;
        NewExpr *newe;
        AllocaExpr *alloce;

        if ((ue = llvm::dyn_cast<UnaryExpr>(node)) != NULL)
            ue->expr = (Expr *)WalkAST(ue->expr, preFunc, postFunc, data);
        else if ((be = llvm::dyn_cast<BinaryExpr>(node)) != NULL) {
            be->arg0 = (Expr *)WalkAST(be->arg0, preFunc, postFunc, data);
            be->arg1 = (Expr *)WalkAST(be->arg1, preFunc, postFunc, data);
        } else if ((ae = llvm::dyn_cast<AssignExpr>(node)) != NULL) {
            ae->lvalue = (Expr *)WalkAST(ae->lvalue, preFunc, postFunc, data);
            ae->rvalue = (Expr *)WalkAST(ae->rvalue, preFunc, postFunc, data);
        } else if ((se = llvm::dyn_cast<SelectExpr>(node)) != NULL) {
            se->test = (Expr *)WalkAST(se->test, preFunc, postFunc, data);
            se->expr1 = (Expr *)WalkAST(se->expr1, preFunc, postFunc, data);
            se->expr2 = (Expr *)WalkAST(se->expr2, preFunc, postFunc, data);
        } else if ((el = llvm::dyn_cast<ExprList>(node)) != NULL) {
            for (unsigned int i = 0; i < el->exprs.size(); ++i)
                el->exprs[i] = (Expr *)WalkAST(el->exprs[i], preFunc, postFunc, data);
        } else if ((fce = llvm::dyn_cast<FunctionCallExpr>(node)) != NULL) {
            fce->func = (Expr *)WalkAST(fce->func, preFunc, postFunc, data);
            fce->args = (ExprList *)WalkAST(fce->args, preFunc, postFunc, data);
            for (int k = 0; k < 3; k++)
                fce->launchCountExpr[k] = (Expr *)WalkAST(fce->launchCountExpr[k], preFunc, postFunc, data);
        } else if ((ie = llvm::dyn_cast<IndexExpr>(node)) != NULL) {
            ie->baseExpr = (Expr *)WalkAST(ie->baseExpr, preFunc, postFunc, data);
            ie->index = (Expr *)WalkAST(ie->index, preFunc, postFunc, data);
        } else if ((me = llvm::dyn_cast<MemberExpr>(node)) != NULL)
            me->expr = (Expr *)WalkAST(me->expr, preFunc, postFunc, data);
        else if ((tce = llvm::dyn_cast<TypeCastExpr>(node)) != NULL)
            tce->expr = (Expr *)WalkAST(tce->expr, preFunc, postFunc, data);
        else if ((re = llvm::dyn_cast<ReferenceExpr>(node)) != NULL)
            re->expr = (Expr *)WalkAST(re->expr, preFunc, postFunc, data);
        else if ((ptrderef = llvm::dyn_cast<PtrDerefExpr>(node)) != NULL)
            ptrderef->expr = (Expr *)WalkAST(ptrderef->expr, preFunc, postFunc, data);
        else if ((refderef = llvm::dyn_cast<RefDerefExpr>(node)) != NULL)
            refderef->expr = (Expr *)WalkAST(refderef->expr, preFunc, postFunc, data);
        else if ((soe = llvm::dyn_cast<SizeOfExpr>(node)) != NULL)
            soe->expr = (Expr *)WalkAST(soe->expr, preFunc, postFunc, data);
        else if ((alloce = llvm::dyn_cast<AllocaExpr>(node)) != NULL)
            alloce->expr = (Expr *)WalkAST(alloce->expr, preFunc, postFunc, data);
        else if ((aoe = llvm::dyn_cast<AddressOfExpr>(node)) != NULL)
            aoe->expr = (Expr *)WalkAST(aoe->expr, preFunc, postFunc, data);
        else if ((newe = llvm::dyn_cast<NewExpr>(node)) != NULL) {
            newe->countExpr = (Expr *)WalkAST(newe->countExpr, preFunc, postFunc, data);
            newe->initExpr = (Expr *)WalkAST(newe->initExpr, preFunc, postFunc, data);
        } else if (llvm::dyn_cast<SymbolExpr>(node) != NULL || llvm::dyn_cast<ConstExpr>(node) != NULL ||
                   llvm::dyn_cast<FunctionSymbolExpr>(node) != NULL || llvm::dyn_cast<SyncExpr>(node) != NULL ||
                   llvm::dyn_cast<NullPointerExpr>(node) != NULL) {
            // nothing to do
        } else
            FATAL("Unhandled expression type in WalkAST().");
    }

    // Call the callback function
    if (postFunc != NULL)
        return postFunc(node, data);
    else
        return node;
}

static ASTNode *lOptimizeNode(ASTNode *node, void *) { return node->Optimize(); }

ASTNode *ispc::Optimize(ASTNode *root) { return WalkAST(root, NULL, lOptimizeNode, NULL); }

Expr *ispc::Optimize(Expr *expr) { return (Expr *)Optimize((ASTNode *)expr); }

Stmt *ispc::Optimize(Stmt *stmt) { return (Stmt *)Optimize((ASTNode *)stmt); }

static ASTNode *lTypeCheckNode(ASTNode *node, void *) { return node->TypeCheck(); }

ASTNode *ispc::TypeCheck(ASTNode *root) { return WalkAST(root, NULL, lTypeCheckNode, NULL); }

Expr *ispc::TypeCheck(Expr *expr) { return (Expr *)TypeCheck((ASTNode *)expr); }

Stmt *ispc::TypeCheck(Stmt *stmt) { return (Stmt *)TypeCheck((ASTNode *)stmt); }

struct CostData {
    CostData() { cost = foreachDepth = 0; }

    int cost;
    int foreachDepth;
};

static bool lCostCallbackPre(ASTNode *node, void *d) {
    CostData *data = (CostData *)d;
    if (llvm::dyn_cast<ForeachStmt>(node) != NULL)
        ++data->foreachDepth;
    if (data->foreachDepth == 0)
        data->cost += node->EstimateCost();
    return true;
}

static ASTNode *lCostCallbackPost(ASTNode *node, void *d) {
    CostData *data = (CostData *)d;
    if (llvm::dyn_cast<ForeachStmt>(node) != NULL)
        --data->foreachDepth;
    return node;
}

int ispc::EstimateCost(ASTNode *root) {
    CostData data;
    WalkAST(root, lCostCallbackPre, lCostCallbackPost, &data);
    return data.cost;
}

/** Given an AST node, check to see if it's safe if we happen to run the
    code for that node with the execution mask all off.
 */
static bool lCheckAllOffSafety(ASTNode *node, void *data) {
    bool *okPtr = (bool *)data;

    FunctionCallExpr *fce;
    if ((fce = llvm::dyn_cast<FunctionCallExpr>(node)) != NULL) {
        if (fce->func == NULL)
            return false;

        const Type *type = fce->func->GetType();
        const PointerType *pt = CastType<PointerType>(type);
        if (pt != NULL)
            type = pt->GetBaseType();
        const FunctionType *ftype = CastType<FunctionType>(type);
        Assert(ftype != NULL);

        if (ftype->isSafe == false) {
            *okPtr = false;
            return false;
        }
    }

    if (llvm::dyn_cast<AssertStmt>(node) != NULL) {
        // While it's fine to run the assert for varying tests, it's not
        // desirable to check an assert on a uniform variable if all of the
        // lanes are off.
        *okPtr = false;
        return false;
    }

    if (llvm::dyn_cast<PrintStmt>(node) != NULL) {
        *okPtr = false;
        return false;
    }

    if (llvm::dyn_cast<NewExpr>(node) != NULL || llvm::dyn_cast<DeleteStmt>(node) != NULL) {
        // We definitely don't want to run the uniform variants of these if
        // the mask is all off.  It's also worth skipping the overhead of
        // executing the varying versions of them in the all-off mask case.
        *okPtr = false;
        return false;
    }

    if (llvm::dyn_cast<ForeachStmt>(node) != NULL || llvm::dyn_cast<ForeachActiveStmt>(node) != NULL ||
        llvm::dyn_cast<ForeachUniqueStmt>(node) != NULL || llvm::dyn_cast<UnmaskedStmt>(node) != NULL) {
        // The various foreach statements also shouldn't be run with an
        // all-off mask.  Since they can re-establish an 'all on' mask,
        // this would be pretty unintuitive.  (More generally, it's
        // possibly a little strange to allow foreach in the presence of
        // any non-uniform control flow...)
        //
        // Similarly, the implementation of foreach_unique assumes as a
        // precondition that the mask won't be all off going into it, so
        // we'll enforce that here...
        *okPtr = false;
        return false;
    }

    BinaryExpr *binaryExpr;
    if ((binaryExpr = llvm::dyn_cast<BinaryExpr>(node)) != NULL) {
        if (binaryExpr->op == BinaryExpr::Mod || binaryExpr->op == BinaryExpr::Div) {
            *okPtr = false;
            return false;
        }
    }
    IndexExpr *ie;
    if ((ie = llvm::dyn_cast<IndexExpr>(node)) != NULL && ie->baseExpr != NULL) {
        const Type *type = ie->baseExpr->GetType();
        if (type == NULL)
            return true;
        if (CastType<ReferenceType>(type) != NULL)
            type = type->GetReferenceTarget();

        ConstExpr *ce = llvm::dyn_cast<ConstExpr>(ie->index);
        if (ce == NULL) {
            // indexing with a variable... -> not safe
            *okPtr = false;
            return false;
        }

        const PointerType *pointerType = CastType<PointerType>(type);
        if (pointerType != NULL) {
            // pointer[index] -> can't be sure -> not safe
            *okPtr = false;
            return false;
        }

        const SequentialType *seqType = CastType<SequentialType>(type);
        Assert(seqType != NULL);
        int nElements = seqType->GetElementCount();
        if (nElements == 0) {
            // Unsized array, so we can't be sure -> not safe
            *okPtr = false;
            return false;
        }

        int32_t indices[ISPC_MAX_NVEC];
        int count = ce->GetValues(indices);
        for (int i = 0; i < count; ++i) {
            if (indices[i] < 0 || indices[i] >= nElements) {
                // Index is out of bounds -> not safe
                *okPtr = false;
                return false;
            }
        }

        // All indices are in-bounds
        return true;
    }

    MemberExpr *me;
    if ((me = llvm::dyn_cast<MemberExpr>(node)) != NULL && me->dereferenceExpr) {
        *okPtr = false;
        return false;
    }

    if (llvm::dyn_cast<PtrDerefExpr>(node) != NULL) {
        *okPtr = false;
        return false;
    }

    /*
      Don't allow turning if/else to straight-line-code if we
      assign to a uniform or post-/pre- increment/decrement.
    */
    AssignExpr *ae;
    if ((ae = llvm::dyn_cast<AssignExpr>(node)) != NULL) {
        if (ae->GetType() && ae->GetType()->IsUniformType()) {
            *okPtr = false;
            return false;
        }
    }

    UnaryExpr *ue;
    if ((ue = llvm::dyn_cast<UnaryExpr>(node)) != NULL &&
        (ue->op == UnaryExpr::PreInc || ue->op == UnaryExpr::PreDec || ue->op == UnaryExpr::PostInc ||
         ue->op == UnaryExpr::PostDec)) {
        if (ue->GetType() && ue->GetType()->IsUniformType()) {
            *okPtr = false;
            return false;
        }
    }

    if (llvm::dyn_cast<SyncExpr>(node) != NULL || llvm::dyn_cast<AllocaExpr>(node) != NULL) {
        *okPtr = false;
        return false;
    }

    return true;
}

bool ispc::SafeToRunWithMaskAllOff(ASTNode *root) {
    bool safe = true;
    WalkAST(root, lCheckAllOffSafety, NULL, &safe);
    return safe;
}
