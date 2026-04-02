/*
  Copyright (c) 2011-2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
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
// Indent

Indent::~Indent() {
    Assert(stack.empty() && "Indent stack is not empty on destruction");
    // Any Print() call must be paired by Done() call
    Assert(printCalls == doneCalls && "AST dump has encountered a bug");
};

void Indent::pushSingle() { stack.push_back(1); }
void Indent::pushList(int i) {
    if (i > 0) {
        stack.push_back(i);
    }
}

void Indent::setNextLabel(std::string s) { label = std::move(s); }

// Print indent and an optional string
void Indent::Print(const char *title) {
    printCalls++;
    Assert(!stack.empty());
    int &top = stack.back();
    Assert(top > 0);

    for (size_t i = 0; i < (stack.size() - 1); i++) {
        if (stack[i] == 0) {
            printf("  ");
        } else {
            printf("| ");
        }
    }

    if (top == 1) {
        printf("`-");
    } else {
        printf("|-");
    }
    top--;

    if (!label.empty()) {
        printf("(%s) ", label.c_str());
        label.clear();
    }

    // An optional string
    if (title != nullptr) {
        printf("%s", title);
    }
}

void Indent::Print(const char *title, const SourcePos &pos) {
    // Same as previous version
    Print(title);
    // Plus source position info
    pos.Print();
}

void Indent::PrintLn(const char *title, const SourcePos &pos) {
    // Same as previous version
    Print(title, pos);
    // Plus end of line
    printf("\n");
}

void Indent::Done() {
    doneCalls++;
    Assert(!stack.empty());
    int &top = stack.back();
    Assert(top >= 0);
    if (top == 0) {
        stack.pop_back();
    }
}

///////////////////////////////////////////////////////////////////////////
// ASTNode

ASTNode::~ASTNode() {}

void ASTNode::Dump() const {
    Indent indent;
    indent.pushSingle();
    Print(indent);
    fflush(stdout);
}

///////////////////////////////////////////////////////////////////////////
// AST

AST::~AST() {
    for (auto p : functions) {
        delete p;
    }
    for (auto p : functionTemplates) {
        delete p;
    }
}

void AST::AddFunction(Symbol *sym, Stmt *code) {
    if (sym == nullptr) {
        return;
    }
    Function *fn = new Function(sym, code);
    sym->parentFunction = fn;
    functions.push_back(fn);
}

void AST::AddFunctionTemplate(TemplateSymbol *templSym, Stmt *code) {
    if (templSym == nullptr || code == nullptr) {
        return;
    }

    FunctionTemplate *funcTempl = new FunctionTemplate(templSym, code);
    templSym->functionTemplate = funcTempl;
    functionTemplates.push_back(funcTempl);
}

void AST::GenerateIR() {
    llvm::TimeTraceScope TimeScope("GenerateIR");
    for (auto fn : functions) {
        fn->GenerateIR();
    }

    for (auto templateFn : functionTemplates) {
        templateFn->GenerateIR();
    }
}

void AST::Print(Globals::ASTDumpKind printKind) const {
    if (printKind == Globals::ASTDumpKind::None) {
        return;
    }

    printf("AST\n");
    Indent indent;

    // Function templates
    int funcsToPrint = 0;
    int funcTemplsToPrint = 0;
    if (printKind == Globals::ASTDumpKind::All) {
        funcTemplsToPrint = functionTemplates.size();
        funcsToPrint = functions.size();
    }

    indent.pushList(funcTemplsToPrint + funcsToPrint);
    // Function templates
    for (unsigned int i = 0; i < functionTemplates.size(); ++i) {
        if (printKind == Globals::ASTDumpKind::All) {
            functionTemplates[i]->Print(indent);
        }
    }

    // Functions
    for (unsigned int i = 0; i < functions.size(); ++i) {
        if (printKind == Globals::ASTDumpKind::All) {
            functions[i]->Print(indent);
        }
    }

    fflush(stdout);
}

///////////////////////////////////////////////////////////////////////////

ASTNode *ispc::WalkAST(ASTNode *node, ASTPreCallBackFunc preFunc, ASTPostCallBackFunc postFunc, void *data) {
    if (node == nullptr) {
        return node;
    }

    // Call the callback function
    if (preFunc != nullptr) {
        if (preFunc(node, data) == false) {
            // The function asked us to not continue recursively, so stop.
            return node;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Handle Statements
    if (llvm::dyn_cast<Stmt>(node) != nullptr) {
        if (ExprStmt *es = llvm::dyn_cast<ExprStmt>(node)) {
            es->expr = (Expr *)WalkAST(es->expr, preFunc, postFunc, data);
        } else if (DeclStmt *ds = llvm::dyn_cast<DeclStmt>(node)) {
            for (unsigned int i = 0; i < ds->vars.size(); ++i) {
                ds->vars[i].init = (Expr *)WalkAST(ds->vars[i].init, preFunc, postFunc, data);
            }
        } else if (IfStmt *is = llvm::dyn_cast<IfStmt>(node)) {
            is->test = (Expr *)WalkAST(is->test, preFunc, postFunc, data);
            is->trueStmts = (Stmt *)WalkAST(is->trueStmts, preFunc, postFunc, data);
            is->falseStmts = (Stmt *)WalkAST(is->falseStmts, preFunc, postFunc, data);
        } else if (DoStmt *dos = llvm::dyn_cast<DoStmt>(node)) {
            dos->testExpr = (Expr *)WalkAST(dos->testExpr, preFunc, postFunc, data);
            dos->bodyStmts = (Stmt *)WalkAST(dos->bodyStmts, preFunc, postFunc, data);
        } else if (ForStmt *fs = llvm::dyn_cast<ForStmt>(node)) {
            fs->init = (Stmt *)WalkAST(fs->init, preFunc, postFunc, data);
            fs->test = (Expr *)WalkAST(fs->test, preFunc, postFunc, data);
            fs->step = (Stmt *)WalkAST(fs->step, preFunc, postFunc, data);
            fs->stmts = (Stmt *)WalkAST(fs->stmts, preFunc, postFunc, data);
        } else if (ForeachStmt *fes = llvm::dyn_cast<ForeachStmt>(node)) {
            for (unsigned int i = 0; i < fes->startExprs.size(); ++i) {
                fes->startExprs[i] = (Expr *)WalkAST(fes->startExprs[i], preFunc, postFunc, data);
            }
            for (unsigned int i = 0; i < fes->endExprs.size(); ++i) {
                fes->endExprs[i] = (Expr *)WalkAST(fes->endExprs[i], preFunc, postFunc, data);
            }
            fes->stmts = (Stmt *)WalkAST(fes->stmts, preFunc, postFunc, data);
        } else if (ForeachActiveStmt *fas = llvm::dyn_cast<ForeachActiveStmt>(node)) {
            fas->stmts = (Stmt *)WalkAST(fas->stmts, preFunc, postFunc, data);
        } else if (ForeachUniqueStmt *fus = llvm::dyn_cast<ForeachUniqueStmt>(node)) {
            fus->expr = (Expr *)WalkAST(fus->expr, preFunc, postFunc, data);
            fus->stmts = (Stmt *)WalkAST(fus->stmts, preFunc, postFunc, data);
        } else if (CaseStmt *cs = llvm::dyn_cast<CaseStmt>(node)) {
            cs->stmts = (Stmt *)WalkAST(cs->stmts, preFunc, postFunc, data);
        } else if (DefaultStmt *defs = llvm::dyn_cast<DefaultStmt>(node)) {
            defs->stmts = (Stmt *)WalkAST(defs->stmts, preFunc, postFunc, data);
        } else if (SwitchStmt *ss = llvm::dyn_cast<SwitchStmt>(node)) {
            ss->expr = (Expr *)WalkAST(ss->expr, preFunc, postFunc, data);
            ss->stmts = (Stmt *)WalkAST(ss->stmts, preFunc, postFunc, data);
        } else if (llvm::dyn_cast<BreakStmt>(node) != nullptr || llvm::dyn_cast<ContinueStmt>(node) != nullptr ||
                   llvm::dyn_cast<GotoStmt>(node) != nullptr) {
            // nothing
        } else if (LabeledStmt *ls = llvm::dyn_cast<LabeledStmt>(node)) {
            ls->stmt = (Stmt *)WalkAST(ls->stmt, preFunc, postFunc, data);
        } else if (ReturnStmt *rs = llvm::dyn_cast<ReturnStmt>(node)) {
            rs->expr = (Expr *)WalkAST(rs->expr, preFunc, postFunc, data);
        } else if (StmtList *sl = llvm::dyn_cast<StmtList>(node)) {
            std::vector<Stmt *> &sls = sl->stmts;
            for (unsigned int i = 0; i < sls.size(); ++i) {
                sls[i] = (Stmt *)WalkAST(sls[i], preFunc, postFunc, data);
            }
        } else if (PrintStmt *ps = llvm::dyn_cast<PrintStmt>(node)) {
            ps->values = (Expr *)WalkAST(ps->values, preFunc, postFunc, data);
        } else if (AssertStmt *as = llvm::dyn_cast<AssertStmt>(node)) {
            as->expr = (Expr *)WalkAST(as->expr, preFunc, postFunc, data);
        } else if (DeleteStmt *dels = llvm::dyn_cast<DeleteStmt>(node)) {
            dels->expr = (Expr *)WalkAST(dels->expr, preFunc, postFunc, data);
        } else if (UnmaskedStmt *ums = llvm::dyn_cast<UnmaskedStmt>(node)) {
            ums->stmts = (Stmt *)WalkAST(ums->stmts, preFunc, postFunc, data);
        } else {
            FATAL("Unhandled statement type in WalkAST()");
        }
    } else {
        ///////////////////////////////////////////////////////////////////////////
        // Handle expressions
        Assert(llvm::dyn_cast<Expr>(node) != nullptr);
        if (UnaryExpr *ue = llvm::dyn_cast<UnaryExpr>(node)) {
            ue->expr = (Expr *)WalkAST(ue->expr, preFunc, postFunc, data);
        } else if (BinaryExpr *be = llvm::dyn_cast<BinaryExpr>(node)) {
            be->arg0 = (Expr *)WalkAST(be->arg0, preFunc, postFunc, data);
            be->arg1 = (Expr *)WalkAST(be->arg1, preFunc, postFunc, data);
        } else if (AssignExpr *ae = llvm::dyn_cast<AssignExpr>(node)) {
            ae->lvalue = (Expr *)WalkAST(ae->lvalue, preFunc, postFunc, data);
            ae->rvalue = (Expr *)WalkAST(ae->rvalue, preFunc, postFunc, data);
        } else if (SelectExpr *se = llvm::dyn_cast<SelectExpr>(node)) {
            se->test = (Expr *)WalkAST(se->test, preFunc, postFunc, data);
            se->expr1 = (Expr *)WalkAST(se->expr1, preFunc, postFunc, data);
            se->expr2 = (Expr *)WalkAST(se->expr2, preFunc, postFunc, data);
        } else if (ExprList *el = llvm::dyn_cast<ExprList>(node)) {
            for (unsigned int i = 0; i < el->exprs.size(); ++i) {
                el->exprs[i] = (Expr *)WalkAST(el->exprs[i], preFunc, postFunc, data);
            }
        } else if (FunctionCallExpr *fce = llvm::dyn_cast<FunctionCallExpr>(node)) {
            fce->func = (Expr *)WalkAST(fce->func, preFunc, postFunc, data);
            fce->args = (ExprList *)WalkAST(fce->args, preFunc, postFunc, data);
            for (int k = 0; k < 3; k++) {
                fce->launchCountExpr[k] = (Expr *)WalkAST(fce->launchCountExpr[k], preFunc, postFunc, data);
            }
        } else if (IndexExpr *ie = llvm::dyn_cast<IndexExpr>(node)) {
            ie->baseExpr = (Expr *)WalkAST(ie->baseExpr, preFunc, postFunc, data);
            ie->index = (Expr *)WalkAST(ie->index, preFunc, postFunc, data);
        } else if (MemberExpr *me = llvm::dyn_cast<MemberExpr>(node)) {
            me->expr = (Expr *)WalkAST(me->expr, preFunc, postFunc, data);
        } else if (TypeCastExpr *tce = llvm::dyn_cast<TypeCastExpr>(node)) {
            tce->expr = (Expr *)WalkAST(tce->expr, preFunc, postFunc, data);
        } else if (ReferenceExpr *re = llvm::dyn_cast<ReferenceExpr>(node)) {
            re->expr = (Expr *)WalkAST(re->expr, preFunc, postFunc, data);
        } else if (PtrDerefExpr *ptrderef = llvm::dyn_cast<PtrDerefExpr>(node)) {
            ptrderef->expr = (Expr *)WalkAST(ptrderef->expr, preFunc, postFunc, data);
        } else if (RefDerefExpr *refderef = llvm::dyn_cast<RefDerefExpr>(node)) {
            refderef->expr = (Expr *)WalkAST(refderef->expr, preFunc, postFunc, data);
        } else if (SizeOfExpr *soe = llvm::dyn_cast<SizeOfExpr>(node)) {
            soe->expr = (Expr *)WalkAST(soe->expr, preFunc, postFunc, data);
        } else if (AllocaExpr *alloce = llvm::dyn_cast<AllocaExpr>(node)) {
            alloce->expr = (Expr *)WalkAST(alloce->expr, preFunc, postFunc, data);
        } else if (AddressOfExpr *aoe = llvm::dyn_cast<AddressOfExpr>(node)) {
            aoe->expr = (Expr *)WalkAST(aoe->expr, preFunc, postFunc, data);
        } else if (NewExpr *newe = llvm::dyn_cast<NewExpr>(node)) {
            newe->countExpr = (Expr *)WalkAST(newe->countExpr, preFunc, postFunc, data);
            newe->initExpr = (Expr *)WalkAST(newe->initExpr, preFunc, postFunc, data);
        } else if (llvm::dyn_cast<SymbolExpr>(node) != nullptr || llvm::dyn_cast<ConstExpr>(node) != nullptr ||
                   llvm::dyn_cast<FunctionSymbolExpr>(node) != nullptr || llvm::dyn_cast<SyncExpr>(node) != nullptr ||
                   llvm::dyn_cast<NullPointerExpr>(node) != nullptr) {
            // nothing to do
        } else {
            FATAL("Unhandled expression type in WalkAST().");
        }
    }

    // Call the callback function
    if (postFunc != nullptr) {
        return postFunc(node, data);
    } else {
        return node;
    }
}

static ASTNode *lTypeCheckNode(ASTNode *node, void *) {
    // Skip if already type-checked
    if (node->IsTypeChecked()) {
        return node;
    }

    if (node->IsTypeCheckInProgress()) {
        // It should not happen but if it did return node without marking it as type-checked to avoid recursion
        Assert(node->IsTypeCheckInProgress());
        return node;
    }

    // Mark that we're starting the type check process for this node
    node->StartTypeCheck();

    ASTNode *result = node->TypeCheck();

    // Mark that we're finished with type checking this node
    node->FinishTypeCheck();

    if (result) {
        // Mark result as type-checked too
        result->SetTypeChecked();
    }
    return result;
}

static ASTNode *lOptimizeNode(ASTNode *node, void *) {
    // Skip if already optimized
    if (node->IsOptimized()) {
        return node;
    }

    if (node->IsOptimizeInProgress()) {
        // It should not happen but if it did return node without marking it as optimized to avoid recursion
        return node;
    }

    Assert(node->IsTypeChecked() && "Node must be type-checked before optimization");

    // Mark that we're starting the optimization process for this node
    node->StartOptimize();

    // Now proceed with optimization
    ASTNode *result = node->Optimize();

    // Mark that we're finished with optimizing this node
    node->FinishOptimize();

    if (result) {
        result->SetOptimized();
    }
    return result;
}

ASTNode *ispc::Optimize(ASTNode *root) { return WalkAST(root, nullptr, lOptimizeNode, nullptr); }

Expr *ispc::Optimize(Expr *expr) { return (Expr *)Optimize((ASTNode *)expr); }

Stmt *ispc::Optimize(Stmt *stmt) { return (Stmt *)Optimize((ASTNode *)stmt); }

ASTNode *ispc::TypeCheck(ASTNode *root) { return WalkAST(root, nullptr, lTypeCheckNode, nullptr); }

Expr *ispc::TypeCheck(Expr *expr) { return (Expr *)TypeCheck((ASTNode *)expr); }

Stmt *ispc::TypeCheck(Stmt *stmt) { return (Stmt *)TypeCheck((ASTNode *)stmt); }

/**
 * Performs type checking followed by optimization on the given node.
 * This encapsulates the common pattern of calling TypeCheck() followed by Optimize().
 */
ASTNode *ispc::TypeCheckAndOptimize(ASTNode *root) {
    if (root == nullptr) {
        return nullptr;
    }

    // First type check
    ASTNode *result = TypeCheck(root);
    if (result == nullptr) {
        return nullptr;
    }

    // Then optimize
    return Optimize(result);
}

/**
 * Convenience version of TypeCheckAndOptimize() for Expr *s
 */
Expr *ispc::TypeCheckAndOptimize(Expr *expr) { return (Expr *)TypeCheckAndOptimize((ASTNode *)expr); }

/**
 * Convenience version of TypeCheckAndOptimize() for Stmt *s
 */
Stmt *ispc::TypeCheckAndOptimize(Stmt *stmt) { return (Stmt *)TypeCheckAndOptimize((ASTNode *)stmt); }

struct CostData {
    CostData() { cost = foreachDepth = 0; }

    int cost;
    int foreachDepth;
};

static bool lCostCallbackPre(ASTNode *node, void *d) {
    CostData *data = (CostData *)d;
    if (llvm::dyn_cast<ForeachStmt>(node) != nullptr) {
        ++data->foreachDepth;
    }
    if (data->foreachDepth == 0) {
        data->cost += node->EstimateCost();
    }
    return true;
}

static ASTNode *lCostCallbackPost(ASTNode *node, void *d) {
    CostData *data = (CostData *)d;
    if (llvm::dyn_cast<ForeachStmt>(node) != nullptr) {
        --data->foreachDepth;
    }
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

    FunctionCallExpr *fce = llvm::dyn_cast<FunctionCallExpr>(node);
    if (fce != nullptr) {
        if (fce->func == nullptr) {
            return false;
        }

        const Type *type = fce->func->GetType();
        const PointerType *pt = CastType<PointerType>(type);
        if (pt != nullptr) {
            type = pt->GetBaseType();
        }
        const FunctionType *ftype = CastType<FunctionType>(type);
        Assert(ftype != nullptr);

        if (ftype->IsSafe() == false) {
            *okPtr = false;
            return false;
        }
    }

    if (llvm::dyn_cast<AssertStmt>(node) != nullptr) {
        // While it's fine to run the assert for varying tests, it's not
        // desirable to check an assert on a uniform variable if all of the
        // lanes are off.
        *okPtr = false;
        return false;
    }

    if (llvm::dyn_cast<PrintStmt>(node) != nullptr) {
        *okPtr = false;
        return false;
    }

    if (llvm::dyn_cast<NewExpr>(node) != nullptr || llvm::dyn_cast<DeleteStmt>(node) != nullptr) {
        // We definitely don't want to run the uniform variants of these if
        // the mask is all off.  It's also worth skipping the overhead of
        // executing the varying versions of them in the all-off mask case.
        *okPtr = false;
        return false;
    }

    if (llvm::dyn_cast<ForeachStmt>(node) != nullptr || llvm::dyn_cast<ForeachActiveStmt>(node) != nullptr ||
        llvm::dyn_cast<ForeachUniqueStmt>(node) != nullptr || llvm::dyn_cast<UnmaskedStmt>(node) != nullptr) {
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

    if (BinaryExpr *binaryExpr = llvm::dyn_cast<BinaryExpr>(node)) {
        if (binaryExpr->op == BinaryExpr::Mod || binaryExpr->op == BinaryExpr::Div) {
            *okPtr = false;
            return false;
        }
    }

    IndexExpr *ie = llvm::dyn_cast<IndexExpr>(node);
    if (ie != nullptr && ie->baseExpr != nullptr) {
        const Type *type = ie->baseExpr->GetType();
        if (type == nullptr) {
            return true;
        }
        if (CastType<ReferenceType>(type) != nullptr) {
            type = type->GetReferenceTarget();
        }

        ConstExpr *ce = llvm::dyn_cast<ConstExpr>(ie->index);
        if (ce == nullptr) {
            // indexing with a variable... -> not safe
            *okPtr = false;
            return false;
        }

        const PointerType *pointerType = CastType<PointerType>(type);
        if (pointerType != nullptr) {
            // pointer[index] -> can't be sure -> not safe
            *okPtr = false;
            return false;
        }

        const SequentialType *seqType = CastType<SequentialType>(type);
        Assert(seqType != nullptr);
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

    if (MemberExpr *me = llvm::dyn_cast<MemberExpr>(node)) {
        if (me->dereferenceExpr) {
            *okPtr = false;
            return false;
        }
    }

    if (llvm::dyn_cast<PtrDerefExpr>(node) != nullptr) {
        *okPtr = false;
        return false;
    }

    /*
      Don't allow turning if/else to straight-line-code if we
      assign to a uniform or post-/pre- increment/decrement.
    */
    if (AssignExpr *ae = llvm::dyn_cast<AssignExpr>(node)) {
        if (ae->GetType() && ae->GetType()->IsUniformType()) {
            *okPtr = false;
            return false;
        }
    }

    if (UnaryExpr *ue = llvm::dyn_cast<UnaryExpr>(node)) {
        if (ue->op == UnaryExpr::PreInc || ue->op == UnaryExpr::PreDec || ue->op == UnaryExpr::PostInc ||
            ue->op == UnaryExpr::PostDec) {
            if (ue->GetType() && ue->GetType()->IsUniformType()) {
                *okPtr = false;
                return false;
            }
        }
    }

    if (llvm::dyn_cast<SyncExpr>(node) != nullptr || llvm::dyn_cast<AllocaExpr>(node) != nullptr) {
        *okPtr = false;
        return false;
    }

    return true;
}

bool ispc::SafeToRunWithMaskAllOff(ASTNode *root) {
    bool safe = true;
    WalkAST(root, lCheckAllOffSafety, nullptr, &safe);
    return safe;
}
