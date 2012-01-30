/*
  Copyright (c) 2011, Intel Corporation
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
    @brief 
*/

#include "ast.h"
#include "expr.h"
#include "func.h"
#include "stmt.h"
#include "sym.h"
#include "util.h"

///////////////////////////////////////////////////////////////////////////
// ASTNode

ASTNode::~ASTNode() {
}


///////////////////////////////////////////////////////////////////////////
// AST

void
AST::AddFunction(Symbol *sym, const std::vector<Symbol *> &args, Stmt *code) {
    if (sym == NULL)
        return;
    functions.push_back(new Function(sym, args, code));
}


void
AST::GenerateIR() {
    for (unsigned int i = 0; i < functions.size(); ++i)
        functions[i]->GenerateIR();
}

///////////////////////////////////////////////////////////////////////////

ASTNode *
WalkAST(ASTNode *node, ASTPreCallBackFunc preFunc, ASTPostCallBackFunc postFunc,
        void *data) {
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
    if (dynamic_cast<Stmt *>(node) != NULL) {
        ExprStmt *es;
        DeclStmt *ds;
        IfStmt *is;
        DoStmt *dos;
        ForStmt *fs;
        ForeachStmt *fes;
        CaseStmt *cs;
        DefaultStmt *defs;
        SwitchStmt *ss;
        ReturnStmt *rs;
        LabeledStmt *ls;
        StmtList *sl;
        PrintStmt *ps;
        AssertStmt *as;
        DeleteStmt *dels;

        if ((es = dynamic_cast<ExprStmt *>(node)) != NULL)
            es->expr = (Expr *)WalkAST(es->expr, preFunc, postFunc, data);
        else if ((ds = dynamic_cast<DeclStmt *>(node)) != NULL) {
            for (unsigned int i = 0; i < ds->vars.size(); ++i)
                ds->vars[i].init = (Expr *)WalkAST(ds->vars[i].init, preFunc, 
                                                   postFunc, data);
        }
        else if ((is = dynamic_cast<IfStmt *>(node)) != NULL) {
            is->test = (Expr *)WalkAST(is->test, preFunc, postFunc, data);
            is->trueStmts = (Stmt *)WalkAST(is->trueStmts, preFunc, 
                                            postFunc, data);
            is->falseStmts = (Stmt *)WalkAST(is->falseStmts, preFunc, 
                                             postFunc, data);
        }
        else if ((dos = dynamic_cast<DoStmt *>(node)) != NULL) {
            dos->testExpr = (Expr *)WalkAST(dos->testExpr, preFunc, 
                                            postFunc, data);
            dos->bodyStmts = (Stmt *)WalkAST(dos->bodyStmts, preFunc, 
                                             postFunc, data);
        }
        else if ((fs = dynamic_cast<ForStmt *>(node)) != NULL) {
            fs->init = (Stmt *)WalkAST(fs->init, preFunc, postFunc, data);
            fs->test = (Expr *)WalkAST(fs->test, preFunc, postFunc, data);
            fs->step = (Stmt *)WalkAST(fs->step, preFunc, postFunc, data);
            fs->stmts = (Stmt *)WalkAST(fs->stmts, preFunc, postFunc, data);
        }
        else if ((fes = dynamic_cast<ForeachStmt *>(node)) != NULL) {
            for (unsigned int i = 0; i < fes->startExprs.size(); ++i)
                fes->startExprs[i] = (Expr *)WalkAST(fes->startExprs[i], preFunc, 
                                                     postFunc, data);
            for (unsigned int i = 0; i < fes->endExprs.size(); ++i)
                fes->endExprs[i] = (Expr *)WalkAST(fes->endExprs[i], preFunc, 
                                                   postFunc, data);
            fes->stmts = (Stmt *)WalkAST(fes->stmts, preFunc, postFunc, data);
        }
        else if ((cs = dynamic_cast<CaseStmt *>(node)) != NULL)
            cs->stmts = (Stmt *)WalkAST(cs->stmts, preFunc, postFunc, data);
        else if ((defs = dynamic_cast<DefaultStmt *>(node)) != NULL)
            defs->stmts = (Stmt *)WalkAST(defs->stmts, preFunc, postFunc, data);
        else if ((ss = dynamic_cast<SwitchStmt *>(node)) != NULL) {
            ss->expr = (Expr *)WalkAST(ss->expr, preFunc, postFunc, data);
            ss->stmts = (Stmt *)WalkAST(ss->stmts, preFunc, postFunc, data);
        }
        else if (dynamic_cast<BreakStmt *>(node) != NULL ||
                 dynamic_cast<ContinueStmt *>(node) != NULL ||
                 dynamic_cast<GotoStmt *>(node) != NULL) {
            // nothing
        }
        else if ((ls = dynamic_cast<LabeledStmt *>(node)) != NULL)
            ls->stmt = (Stmt *)WalkAST(ls->stmt, preFunc, postFunc, data);
        else if ((rs = dynamic_cast<ReturnStmt *>(node)) != NULL)
            rs->val = (Expr *)WalkAST(rs->val, preFunc, postFunc, data);
        else if ((sl = dynamic_cast<StmtList *>(node)) != NULL) {
            std::vector<Stmt *> &sls = sl->stmts;
            for (unsigned int i = 0; i < sls.size(); ++i)
                sls[i] = (Stmt *)WalkAST(sls[i], preFunc, postFunc, data);
        }
        else if ((ps = dynamic_cast<PrintStmt *>(node)) != NULL)
            ps->values = (Expr *)WalkAST(ps->values, preFunc, postFunc, data);
        else if ((as = dynamic_cast<AssertStmt *>(node)) != NULL)
            as->expr = (Expr *)WalkAST(as->expr, preFunc, postFunc, data);
        else if ((dels = dynamic_cast<DeleteStmt *>(node)) != NULL)
            dels->expr = (Expr *)WalkAST(dels->expr, preFunc, postFunc, data);
        else
            FATAL("Unhandled statement type in WalkAST()");
    }
    else {
        ///////////////////////////////////////////////////////////////////////////
        // Handle expressions
        Assert(dynamic_cast<Expr *>(node) != NULL);
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
        DereferenceExpr *dre;
        SizeOfExpr *soe;
        AddressOfExpr *aoe;
        NewExpr *newe;

        if ((ue = dynamic_cast<UnaryExpr *>(node)) != NULL)
            ue->expr = (Expr *)WalkAST(ue->expr, preFunc, postFunc, data);
        else if ((be = dynamic_cast<BinaryExpr *>(node)) != NULL) {
            be->arg0 = (Expr *)WalkAST(be->arg0, preFunc, postFunc, data);
            be->arg1 = (Expr *)WalkAST(be->arg1, preFunc, postFunc, data);
        }
        else if ((ae = dynamic_cast<AssignExpr *>(node)) != NULL) {
            ae->lvalue = (Expr *)WalkAST(ae->lvalue, preFunc, postFunc, data);
            ae->rvalue = (Expr *)WalkAST(ae->rvalue, preFunc, postFunc, data);
        }
        else if ((se = dynamic_cast<SelectExpr *>(node)) != NULL) {
            se->test = (Expr *)WalkAST(se->test, preFunc, postFunc, data);
            se->expr1 = (Expr *)WalkAST(se->expr1, preFunc, postFunc, data);
            se->expr2 = (Expr *)WalkAST(se->expr2, preFunc, postFunc, data);
        }
        else if ((el = dynamic_cast<ExprList *>(node)) != NULL) {
            for (unsigned int i = 0; i < el->exprs.size(); ++i)
                el->exprs[i] = (Expr *)WalkAST(el->exprs[i], preFunc, 
                                               postFunc, data);
        }
        else if ((fce = dynamic_cast<FunctionCallExpr *>(node)) != NULL) {
            fce->func = (Expr *)WalkAST(fce->func, preFunc, postFunc, data);
            fce->args = (ExprList *)WalkAST(fce->args, preFunc, postFunc, data);
            fce->launchCountExpr = (Expr *)WalkAST(fce->launchCountExpr, preFunc,
                                                   postFunc, data);
        }
        else if ((ie = dynamic_cast<IndexExpr *>(node)) != NULL) {
            ie->baseExpr = (Expr *)WalkAST(ie->baseExpr, preFunc, postFunc, data);
            ie->index = (Expr *)WalkAST(ie->index, preFunc, postFunc, data);
        }
        else if ((me = dynamic_cast<MemberExpr *>(node)) != NULL)
            me->expr = (Expr *)WalkAST(me->expr, preFunc, postFunc, data);
        else if ((tce = dynamic_cast<TypeCastExpr *>(node)) != NULL)
            tce->expr = (Expr *)WalkAST(tce->expr, preFunc, postFunc, data);
        else if ((re = dynamic_cast<ReferenceExpr *>(node)) != NULL)
            re->expr = (Expr *)WalkAST(re->expr, preFunc, postFunc, data);
        else if ((dre = dynamic_cast<DereferenceExpr *>(node)) != NULL)
            dre->expr = (Expr *)WalkAST(dre->expr, preFunc, postFunc, data);
        else if ((soe = dynamic_cast<SizeOfExpr *>(node)) != NULL)
            soe->expr = (Expr *)WalkAST(soe->expr, preFunc, postFunc, data);
        else if ((aoe = dynamic_cast<AddressOfExpr *>(node)) != NULL)
            aoe->expr = (Expr *)WalkAST(aoe->expr, preFunc, postFunc, data);
        else if ((newe = dynamic_cast<NewExpr *>(node)) != NULL) {
            newe->countExpr = (Expr *)WalkAST(newe->countExpr, preFunc, 
                                              postFunc, data);
            newe->initExpr = (Expr *)WalkAST(newe->initExpr, preFunc, 
                                             postFunc, data);
        }
        else if (dynamic_cast<SymbolExpr *>(node) != NULL ||
                 dynamic_cast<ConstExpr *>(node) != NULL ||
                 dynamic_cast<FunctionSymbolExpr *>(node) != NULL ||
                 dynamic_cast<SyncExpr *>(node) != NULL ||
                 dynamic_cast<NullPointerExpr *>(node) != NULL) {
            // nothing to do 
        }
        else 
            FATAL("Unhandled expression type in WalkAST().");
    }

    // Call the callback function
    if (postFunc != NULL)
        return postFunc(node, data);
    else
        return node;
}


static ASTNode *
lOptimizeNode(ASTNode *node, void *) {
    return node->Optimize();
}


ASTNode *
Optimize(ASTNode *root) {
    return WalkAST(root, NULL, lOptimizeNode, NULL);
}


Expr *
Optimize(Expr *expr) {
    return (Expr *)Optimize((ASTNode *)expr);
}


Stmt *
Optimize(Stmt *stmt) {
    return (Stmt *)Optimize((ASTNode *)stmt);
}


static ASTNode *
lTypeCheckNode(ASTNode *node, void *) {
    return node->TypeCheck();
}


ASTNode *
TypeCheck(ASTNode *root) {
    return WalkAST(root, NULL, lTypeCheckNode, NULL);
}


Expr *
TypeCheck(Expr *expr) {
    return (Expr *)TypeCheck((ASTNode *)expr);
}


Stmt *
TypeCheck(Stmt *stmt) {
    return (Stmt *)TypeCheck((ASTNode *)stmt);
}


static bool
lCostCallback(ASTNode *node, void *c) {
    int *cost = (int *)c;
    *cost += node->EstimateCost();
    return true;
}


int
EstimateCost(ASTNode *root) {
    int cost = 0;
    WalkAST(root, lCostCallback, NULL, &cost);
    return cost;
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

    if (dynamic_cast<NewExpr *>(node) != NULL ||
        dynamic_cast<DeleteStmt *>(node) != NULL) {
        // We definitely don't want to run the uniform variants of these if
        // the mask is all off.  It's also worth skipping the overhead of
        // executing the varying versions of them in the all-off mask case.
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
        return true;
    }

    MemberExpr *me;
    if ((me = dynamic_cast<MemberExpr *>(node)) != NULL &&
        me->dereferenceExpr) {
        *okPtr = false;
        return false;
    }

    DereferenceExpr *de;
    if ((de = dynamic_cast<DereferenceExpr *>(node)) != NULL) {
        const Type *exprType = de->expr->GetType();
        if (dynamic_cast<const PointerType *>(exprType) != NULL) {
            *okPtr = false;
            return false;
        }
    }

    return true;
}


bool
SafeToRunWithMaskAllOff(ASTNode *root) {
    bool safe = true;
    WalkAST(root, lCheckAllOffSafety, NULL, &safe);
    return safe;
}
