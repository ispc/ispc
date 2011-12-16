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

void
WalkAST(ASTNode *node, ASTCallBackFunc preFunc, ASTCallBackFunc postFunc,
        void *data) {
    if (node == NULL)
        return;

    // Call the callback function
    if (preFunc != NULL) {
        if (preFunc(node, data) == false)
            // The function asked us to not continue recursively, so stop.
            return;
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
        ReturnStmt *rs;
        StmtList *sl;
        PrintStmt *ps;
        AssertStmt *as;

        if ((es = dynamic_cast<ExprStmt *>(node)) != NULL)
            WalkAST(es->expr, preFunc, postFunc, data);
        else if ((ds = dynamic_cast<DeclStmt *>(node)) != NULL) {
            for (unsigned int i = 0; i < ds->vars.size(); ++i)
                WalkAST(ds->vars[i].init, preFunc, postFunc, data);
        }
        else if ((is = dynamic_cast<IfStmt *>(node)) != NULL) {
            WalkAST(is->test, preFunc, postFunc, data);
            WalkAST(is->trueStmts, preFunc, postFunc, data);
            WalkAST(is->falseStmts, preFunc, postFunc, data);
        }
        else if ((dos = dynamic_cast<DoStmt *>(node)) != NULL) {
            WalkAST(dos->testExpr, preFunc, postFunc, data);
            WalkAST(dos->bodyStmts, preFunc, postFunc, data);
        }
        else if ((fs = dynamic_cast<ForStmt *>(node)) != NULL) {
            WalkAST(fs->init, preFunc, postFunc, data);
            WalkAST(fs->test, preFunc, postFunc, data);
            WalkAST(fs->step, preFunc, postFunc, data);
            WalkAST(fs->stmts, preFunc, postFunc, data);
        }
        else if ((fes = dynamic_cast<ForeachStmt *>(node)) != NULL) {
            for (unsigned int i = 0; i < fes->startExprs.size(); ++i)
                WalkAST(fes->startExprs[i], preFunc, postFunc, data);
            for (unsigned int i = 0; i < fes->endExprs.size(); ++i)
                WalkAST(fes->endExprs[i], preFunc, postFunc, data);
            WalkAST(fes->stmts, preFunc, postFunc, data);
        }
        else if (dynamic_cast<BreakStmt *>(node) != NULL ||
                 dynamic_cast<ContinueStmt *>(node) != NULL) {
            // nothing 
        }
        else if ((rs = dynamic_cast<ReturnStmt *>(node)) != NULL)
            WalkAST(rs->val, preFunc, postFunc, data);
        else if ((sl = dynamic_cast<StmtList *>(node)) != NULL) {
            const std::vector<Stmt *> &sls = sl->GetStatements();
            for (unsigned int i = 0; i < sls.size(); ++i)
                WalkAST(sls[i], preFunc, postFunc, data);
        }
        else if ((ps = dynamic_cast<PrintStmt *>(node)) != NULL)
            WalkAST(ps->values, preFunc, postFunc, data);
        else if ((as = dynamic_cast<AssertStmt *>(node)) != NULL)
            return WalkAST(as->expr, preFunc, postFunc, data);
        else
            FATAL("Unhandled statement type in WalkAST()");
    }
    else {
        ///////////////////////////////////////////////////////////////////////////
        // Handle expressions
        assert(dynamic_cast<Expr *>(node) != NULL);
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

        if ((ue = dynamic_cast<UnaryExpr *>(node)) != NULL)
            WalkAST(ue->expr, preFunc, postFunc, data);
        else if ((be = dynamic_cast<BinaryExpr *>(node)) != NULL) {
            WalkAST(be->arg0, preFunc, postFunc, data);
            WalkAST(be->arg1, preFunc, postFunc, data);
        }
        else if ((ae = dynamic_cast<AssignExpr *>(node)) != NULL) {
            WalkAST(ae->lvalue, preFunc, postFunc, data);
            WalkAST(ae->rvalue, preFunc, postFunc, data);
        }
        else if ((se = dynamic_cast<SelectExpr *>(node)) != NULL) {
            WalkAST(se->test, preFunc, postFunc, data);
            WalkAST(se->expr1, preFunc, postFunc, data);
            WalkAST(se->expr2, preFunc, postFunc, data);
        }
        else if ((el = dynamic_cast<ExprList *>(node)) != NULL) {
            for (unsigned int i = 0; i < el->exprs.size(); ++i)
                WalkAST(el->exprs[i], preFunc, postFunc, data);
        }
        else if ((fce = dynamic_cast<FunctionCallExpr *>(node)) != NULL) {
            WalkAST(fce->func, preFunc, postFunc, data);
            WalkAST(fce->args, preFunc, postFunc, data);
            WalkAST(fce->launchCountExpr, preFunc, postFunc, data);
        }
        else if ((ie = dynamic_cast<IndexExpr *>(node)) != NULL) {
            WalkAST(ie->baseExpr, preFunc, postFunc, data);
            WalkAST(ie->index, preFunc, postFunc, data);
        }
        else if ((me = dynamic_cast<MemberExpr *>(node)) != NULL)
            WalkAST(me->expr, preFunc, postFunc, data);
        else if ((tce = dynamic_cast<TypeCastExpr *>(node)) != NULL)
            return WalkAST(tce->expr, preFunc, postFunc, data);
        else if ((re = dynamic_cast<ReferenceExpr *>(node)) != NULL)
            return WalkAST(re->expr, preFunc, postFunc, data);
        else if ((dre = dynamic_cast<DereferenceExpr *>(node)) != NULL)
            return WalkAST(dre->expr, preFunc, postFunc, data);
        else if ((soe = dynamic_cast<SizeOfExpr *>(node)) != NULL)
            return WalkAST(soe->expr, preFunc, postFunc, data);
        else if ((aoe = dynamic_cast<AddressOfExpr *>(node)) != NULL)
            WalkAST(aoe->expr, preFunc, postFunc, data);
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
        postFunc(node, data);
}
