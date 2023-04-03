/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file stmt.h
    @brief File with declarations for classes related to statements in the language
*/

#pragma once

#include "ast.h"
#include "ispc.h"

namespace ispc {

/** @brief Interface class for statements in the ispc language.

    This abstract base-class encapsulates methods that AST nodes for
    statements in the language must implement.
 */
class Stmt : public ASTNode {
  public:
    Stmt(SourcePos p, unsigned scid) : ASTNode(p, scid) {}

    static inline bool classof(Stmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() > MaxExprID; }

    /** Emit LLVM IR for the statement, using the FunctionEmitContext to create the
        necessary instructions.
     */
    virtual void EmitCode(FunctionEmitContext *ctx) const = 0;

    // Redeclare these methods with Stmt * return values, rather than
    // ASTNode *s, as in the original ASTNode declarations of them.  We'll
    // also provide a default implementation of Optimize(), since most
    // Stmts don't have anything to do here.
    virtual Stmt *Optimize();
    virtual Stmt *TypeCheck() = 0;
    virtual Stmt *Instantiate(TemplateInstantiation &templInst) const = 0;

    virtual void SetLoopAttribute(std::pair<Globals::pragmaUnrollType, int>);
};

/** @brief Statement representing a single expression */
class ExprStmt : public Stmt {
  public:
    ExprStmt(Expr *expr, SourcePos pos);

    static inline bool classof(ExprStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == ExprStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;
    ExprStmt *Instantiate(TemplateInstantiation &templInst) const;

    Expr *expr;
};

struct VariableDeclaration {
    VariableDeclaration(Symbol *s = nullptr, Expr *i = nullptr) {
        sym = s;
        init = i;
    }
    Symbol *sym;
    Expr *init;
};

/** @brief Statement representing a single declaration (which in turn may declare
    a number of variables. */
class DeclStmt : public Stmt {
  public:
    DeclStmt(const std::vector<VariableDeclaration> &v, SourcePos pos);

    static inline bool classof(DeclStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == DeclStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *Optimize();
    Stmt *TypeCheck();
    int EstimateCost() const;
    DeclStmt *Instantiate(TemplateInstantiation &templInst) const;

    std::vector<VariableDeclaration> vars;
};

/** @brief Statement representing a single if statement, possibly with an
    else clause. */
class IfStmt : public Stmt {
  public:
    IfStmt(Expr *testExpr, Stmt *trueStmts, Stmt *falseStmts, bool doAllCheck, SourcePos pos);

    static inline bool classof(IfStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == IfStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;
    IfStmt *Instantiate(TemplateInstantiation &templInst) const;

    // @todo these are only public for lHasVaryingBreakOrContinue(); would
    // be nice to clean that up...
    /** Expression giving the 'if' test. */
    Expr *test;
    /** Statements to run if the 'if' test returns a true value */
    Stmt *trueStmts;
    /** Statements to run if the 'if' test returns a false value */
    Stmt *falseStmts;

  private:
    /** This value records if this was a 'coherent' if statement in the
        source and thus, if the emitted code should check to see if all
        active program instances want to follow just one of the 'true' or
        'false' blocks. */
    const bool doAllCheck;

    void emitMaskedTrueAndFalse(FunctionEmitContext *ctx, llvm::Value *oldMask, llvm::Value *test) const;
    void emitVaryingIf(FunctionEmitContext *ctx, llvm::Value *test) const;
    void emitMaskAllOn(FunctionEmitContext *ctx, llvm::Value *test, llvm::BasicBlock *bDone) const;
    void emitMaskMixed(FunctionEmitContext *ctx, llvm::Value *oldMask, llvm::Value *test,
                       llvm::BasicBlock *bDone) const;
};

/** @brief Statement implementation representing a 'do' statement in the
    program.
 */
class DoStmt : public Stmt {
  public:
    DoStmt(Expr *testExpr, Stmt *bodyStmts, bool doCoherentCheck, SourcePos pos);

    static inline bool classof(DoStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == DoStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();

    std::pair<Globals::pragmaUnrollType, int> loopAttribute =
        std::pair<Globals::pragmaUnrollType, int>(Globals::pragmaUnrollType::none, -1);
    void SetLoopAttribute(std::pair<Globals::pragmaUnrollType, int>);
    int EstimateCost() const;
    DoStmt *Instantiate(TemplateInstantiation &templInst) const;

    Expr *testExpr;
    Stmt *bodyStmts;
    const bool doCoherentCheck;
};

/** @brief Statement implementation for 'for' loops (as well as for 'while'
    loops).
 */
class ForStmt : public Stmt {
  public:
    ForStmt(Stmt *initializer, Expr *testExpr, Stmt *stepStatements, Stmt *bodyStatements, bool doCoherentCheck,
            SourcePos pos);

    static inline bool classof(ForStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == ForStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();

    std::pair<Globals::pragmaUnrollType, int> loopAttribute =
        std::pair<Globals::pragmaUnrollType, int>(Globals::pragmaUnrollType::none, -1);
    void SetLoopAttribute(std::pair<Globals::pragmaUnrollType, int>);
    int EstimateCost() const;
    ForStmt *Instantiate(TemplateInstantiation &templInst) const;

    /** 'for' statment initializer; may be nullptr, indicating no intitializer */
    Stmt *init;
    /** expression that returns a value indicating whether the loop should
        continue for the next iteration */
    Expr *test;
    /** Statements to run at the end of the loop for the loop step, before
        the test expression is evaluated. */
    Stmt *step;
    /** Loop body statements */
    Stmt *stmts;
    const bool doCoherentCheck;
};

/** @brief Statement implementation for a break statement in the
    program. */
class BreakStmt : public Stmt {
  public:
    BreakStmt(SourcePos pos);

    static inline bool classof(BreakStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == BreakStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;
    BreakStmt *Instantiate(TemplateInstantiation &templInst) const;
};

/** @brief Statement implementation for a continue statement in the
    program. */
class ContinueStmt : public Stmt {
  public:
    ContinueStmt(SourcePos pos);

    static inline bool classof(ContinueStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == ContinueStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;
    ContinueStmt *Instantiate(TemplateInstantiation &templInst) const;
};

/** @brief Statement implementation for parallel 'foreach' loops.
 */
class ForeachStmt : public Stmt {
  public:
    ForeachStmt(const std::vector<Symbol *> &loopVars, const std::vector<Expr *> &startExprs,
                const std::vector<Expr *> &endExprs, Stmt *bodyStatements, bool tiled, SourcePos pos);

    static inline bool classof(ForeachStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == ForeachStmtID; }

#ifdef ISPC_XE_ENABLED
    void EmitCodeForXe(FunctionEmitContext *ctx) const;
#endif
    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    std::pair<Globals::pragmaUnrollType, int> loopAttribute =
        std::pair<Globals::pragmaUnrollType, int>(Globals::pragmaUnrollType::none, -1);
    void SetLoopAttribute(std::pair<Globals::pragmaUnrollType, int>);
    int EstimateCost() const;
    ForeachStmt *Instantiate(TemplateInstantiation &templInst) const;

    std::vector<Symbol *> dimVariables;
    std::vector<Expr *> startExprs;
    std::vector<Expr *> endExprs;
    bool isTiled;
    Stmt *stmts;
};

/** Iteration over each executing program instance.
 */
class ForeachActiveStmt : public Stmt {
  public:
    ForeachActiveStmt(Symbol *iterSym, Stmt *stmts, SourcePos pos);

    static inline bool classof(ForeachActiveStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == ForeachActiveStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    std::pair<Globals::pragmaUnrollType, int> loopAttribute =
        std::pair<Globals::pragmaUnrollType, int>(Globals::pragmaUnrollType::none, -1);
    void SetLoopAttribute(std::pair<Globals::pragmaUnrollType, int>);
    int EstimateCost() const;
    ForeachActiveStmt *Instantiate(TemplateInstantiation &templInst) const;

    Symbol *sym;
    Stmt *stmts;
};

/** Parallel iteration over each unique value in the given (varying)
    expression.
 */
class ForeachUniqueStmt : public Stmt {
  public:
    ForeachUniqueStmt(const char *iterName, Expr *expr, Stmt *stmts, SourcePos pos);
    ForeachUniqueStmt(Symbol *sym, Expr *expr, Stmt *stmts, SourcePos pos);

    static inline bool classof(ForeachUniqueStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == ForeachUniqueStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    std::pair<Globals::pragmaUnrollType, int> loopAttribute =
        std::pair<Globals::pragmaUnrollType, int>(Globals::pragmaUnrollType::none, -1);
    void SetLoopAttribute(std::pair<Globals::pragmaUnrollType, int>);
    int EstimateCost() const;
    ForeachUniqueStmt *Instantiate(TemplateInstantiation &templInst) const;

    Symbol *sym;
    Expr *expr;
    Stmt *stmts;
};

/**
 */
class UnmaskedStmt : public Stmt {
  public:
    UnmaskedStmt(Stmt *stmt, SourcePos pos);

    static inline bool classof(UnmaskedStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == UnmaskedStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;
    UnmaskedStmt *Instantiate(TemplateInstantiation &templInst) const;

    Stmt *stmts;
};

/** @brief Statement implementation for a 'return' statement in the
    program. */
class ReturnStmt : public Stmt {
  public:
    ReturnStmt(Expr *e, SourcePos p);

    static inline bool classof(ReturnStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == ReturnStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;
    ReturnStmt *Instantiate(TemplateInstantiation &templInst) const;

    Expr *expr;
};

/** Statement corresponding to a "case" label in the program.  In addition
    to the value associated with the "case", this statement also stores the
    statements following it. */
class CaseStmt : public Stmt {
  public:
    CaseStmt(int value, Stmt *stmt, SourcePos pos);

    static inline bool classof(CaseStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == CaseStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;
    CaseStmt *Instantiate(TemplateInstantiation &templInst) const;

    /** Integer value after the "case" statement */
    const int value;
    Stmt *stmts;
};

/** Statement for a "default" label (as would be found inside a "switch"
    statement). */
class DefaultStmt : public Stmt {
  public:
    DefaultStmt(Stmt *stmt, SourcePos pos);

    static inline bool classof(DefaultStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == DefaultStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;
    DefaultStmt *Instantiate(TemplateInstantiation &templInst) const;

    Stmt *stmts;
};

/** A "switch" statement in the program. */
class SwitchStmt : public Stmt {
  public:
    SwitchStmt(Expr *expr, Stmt *stmts, SourcePos pos);

    static inline bool classof(SwitchStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == SwitchStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;
    SwitchStmt *Instantiate(TemplateInstantiation &templInst) const;

    /** Expression that is used to determine which label to jump to. */
    Expr *expr;
    /** Statement block after the "switch" expression. */
    Stmt *stmts;
};

/** A "goto" in an ispc program. */
class GotoStmt : public Stmt {
  public:
    GotoStmt(const char *label, SourcePos gotoPos, SourcePos idPos);

    static inline bool classof(GotoStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == GotoStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *Optimize();
    Stmt *TypeCheck();
    int EstimateCost() const;
    GotoStmt *Instantiate(TemplateInstantiation &templInst) const;

    /** Name of the label to jump to when the goto is executed. */
    std::string label;
    SourcePos identifierPos;
};

/** Statement corresponding to a label (as would be used as a goto target)
    in the program. */
class LabeledStmt : public Stmt {
  public:
    LabeledStmt(const char *label, Stmt *stmt, SourcePos p);

    static inline bool classof(LabeledStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == LabeledStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *Optimize();
    Stmt *TypeCheck();
    int EstimateCost() const;
    LabeledStmt *Instantiate(TemplateInstantiation &templInst) const;

    /** Name of the label. */
    std::string name;
    /** Statements following the label. */
    Stmt *stmt;
};

/** @brief Representation of a list of statements in the program.
 */
class StmtList : public Stmt {
  public:
    StmtList(SourcePos p) : Stmt(p, StmtListID) {}

    static inline bool classof(StmtList const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == StmtListID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;
    StmtList *Instantiate(TemplateInstantiation &templInst) const;

    void Add(Stmt *s) {
        if (s)
            stmts.push_back(s);
    }

    std::vector<Stmt *> stmts;
};

/** @brief Representation of a print() statement in the program.

    It's currently necessary to have a special statement type for print()
    since strings aren't supported as first-class types in the language,
    but we need to be able to pass a formatting string as the first
    argument to print().  We also need this to be a variable argument
    function, which also isn't supported.  Representing print() as a
    statement lets us work around both of those ugly little issues...
  */
class PrintStmt : public Stmt {
  public:
    PrintStmt(const std::string &f, Expr *v, SourcePos p);

    static inline bool classof(PrintStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == PrintStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;
    PrintStmt *Instantiate(TemplateInstantiation &templInst) const;

    std::vector<llvm::Value *> getDoPrintArgs(FunctionEmitContext *ctx) const;
    std::vector<llvm::Value *> getPrintImplArgs(FunctionEmitContext *ctx) const;
#ifdef ISPC_XE_ENABLED
    std::vector<llvm::Value *> getDoPrintCMArgs(FunctionEmitContext *ctx) const;
    void emitCode4LZ(FunctionEmitContext *ctx) const;
#endif // ISPC_XE_ENABLED

    enum {
        FORMAT_IDX,                // the format string
        TYPES_IDX,                 // a string encoding the types of the values being printed,
                                   // one character per value
        WIDTH_IDX,                 // the number of running program instances (i.e. the target's
                                   // vector width)
        MASK_IDX,                  // the current lane mask
        ARGS_IDX,                  // a pointer to an array of pointers to the values to be printed
        STD_NUM_IDX = ARGS_IDX + 1 // number of arguments of __do_print
    };

    /** Format string for the print() statement. */
    const std::string format;
    /** This holds the arguments passed to the print() statement.  If more
        than one was provided, this will be an ExprList. */
    Expr *values;
};

/** @brief Representation of an assert statement in the program.

    Like print() above, since we don't have strings as first-class types in
    the language, we need to do some gymnastics to support it.  Like
    assert() in C, assert() checks the given condition and prints an error
    and calls abort if the condition fails.  For varying conditions, the
    assert triggers if it's true for any of the program instances.
*/
class AssertStmt : public Stmt {
  public:
    AssertStmt(const std::string &msg, Expr *e, SourcePos p);

    static inline bool classof(AssertStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == AssertStmtID; }

    void EmitAssertCode(FunctionEmitContext *ctx, const Type *type) const;
    void EmitAssumeCode(FunctionEmitContext *ctx, const Type *type) const;
    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;
    AssertStmt *Instantiate(TemplateInstantiation &templInst) const;

    /** Message to print if the assertion fails. */
    const std::string message;
    /** The expression to be evaluated (that is asserted to be true). */
    Expr *expr;
};

/** Representation of a delete statement in the program.
 */
class DeleteStmt : public Stmt {
  public:
    DeleteStmt(Expr *e, SourcePos p);

    static inline bool classof(DeleteStmt const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == DeleteStmtID; }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(Indent &indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;
    DeleteStmt *Instantiate(TemplateInstantiation &templInst) const;

    /** Expression that gives the pointer value to be deleted. */
    Expr *expr;
};

extern Stmt *CreateForeachActiveStmt(Symbol *iterSym, Stmt *stmts, SourcePos pos);
} // namespace ispc
