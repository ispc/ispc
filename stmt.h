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

/** @file stmt.h
    @brief File with declarations for classes related to statements in the language
*/

#ifndef ISPC_STMT_H
#define ISPC_STMT_H 1

#include "ispc.h"
#include "ast.h"

/** @brief Interface class for statements in the ispc language.

    This abstract base-class encapsulates methods that AST nodes for
    statements in the language must implement.
 */
class Stmt : public ASTNode {
public:
    Stmt(SourcePos p) : ASTNode(p) { }

    /** Emit LLVM IR for the statement, using the FunctionEmitContext to create the
        necessary instructions.
     */
    virtual void EmitCode(FunctionEmitContext *ctx) const = 0;

    /** Print a representation of the statement (and any children AST
        nodes) to standard output.  This method is used for debuggins. */
    virtual void Print(int indent) const = 0;

    // Redeclare these methods with Stmt * return values, rather than
    // ASTNode *s, as in the original ASTNode declarations of them.  We'll
    // also provide a default implementation of Optimize(), since most
    // Stmts don't have anything to do here.
    virtual Stmt *Optimize();
    virtual Stmt *TypeCheck() = 0;
};


/** @brief Statement representing a single expression */
class ExprStmt : public Stmt {
public:
    ExprStmt(Expr *expr, SourcePos pos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

    Expr *expr;
};


struct VariableDeclaration {
    VariableDeclaration(Symbol *s = NULL, Expr *i = NULL) { 
        sym = s; init = i; 
    }
    Symbol *sym;
    Expr *init;
};

/** @brief Statement representing a single declaration (which in turn may declare
    a number of variables. */
class DeclStmt : public Stmt {
public:
    DeclStmt(const std::vector<VariableDeclaration> &v, SourcePos pos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *Optimize();
    Stmt *TypeCheck();
    int EstimateCost() const;

    std::vector<VariableDeclaration> vars;
};


/** @brief Statement representing a single if statement, possibly with an
    else clause. */
class IfStmt : public Stmt {
public:
    IfStmt(Expr *testExpr, Stmt *trueStmts, Stmt *falseStmts,
           bool doAllCheck, SourcePos pos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

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

    void emitMaskedTrueAndFalse(FunctionEmitContext *ctx, llvm::Value *oldMask, 
                                llvm::Value *test) const;
    void emitVaryingIf(FunctionEmitContext *ctx, llvm::Value *test) const;
    void emitMaskAllOn(FunctionEmitContext *ctx,
                       llvm::Value *test, llvm::BasicBlock *bDone) const;
    void emitMaskMixed(FunctionEmitContext *ctx, llvm::Value *oldMask, 
                       llvm::Value *test, llvm::BasicBlock *bDone) const;
};


/** @brief Statement implementation representing a 'do' statement in the
    program.
 */
class DoStmt : public Stmt {
public:
    DoStmt(Expr *testExpr, Stmt *bodyStmts, bool doCoherentCheck, 
           SourcePos pos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

    Expr *testExpr;
    Stmt *bodyStmts;
    const bool doCoherentCheck;
};


/** @brief Statement implementation for 'for' loops (as well as for 'while'
    loops).
 */
class ForStmt : public Stmt {
public:
    ForStmt(Stmt *initializer, Expr *testExpr, Stmt *stepStatements,
            Stmt *bodyStatements, bool doCoherentCheck, SourcePos pos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

    /** 'for' statment initializer; may be NULL, indicating no intitializer */
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


/** @brief Statement implementation for a break or 'coherent' break
    statement in the program. */
class BreakStmt : public Stmt {
public:
    BreakStmt(bool doCoherenceCheck, SourcePos pos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

private:
    /** This indicates whether the generated code will check to see if no
        more program instances are currently running after the break, in
        which case the code can have a jump to the end of the current
        loop. */
    const bool doCoherenceCheck;
};


/** @brief Statement implementation for a continue or 'coherent' continue
    statement in the program. */
class ContinueStmt : public Stmt {
public:
    ContinueStmt(bool doCoherenceCheck, SourcePos pos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

private:
    /** This indicates whether the generated code will check to see if no
        more program instances are currently running after the continue, in
        which case the code can have a jump to the end of the current
        loop. */
    const bool doCoherenceCheck;
};


/** @brief Statement implementation for parallel 'foreach' loops.
 */
class ForeachStmt : public Stmt {
public:
    ForeachStmt(const std::vector<Symbol *> &loopVars, 
                const std::vector<Expr *> &startExprs, 
                const std::vector<Expr *> &endExprs, 
                Stmt *bodyStatements, bool tiled, SourcePos pos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

    std::vector<Symbol *> dimVariables;
    std::vector<Expr *> startExprs;
    std::vector<Expr *> endExprs;
    bool isTiled;
    Stmt *stmts;
};



/** @brief Statement implementation for a 'return' or 'coherent' return
    statement in the program. */
class ReturnStmt : public Stmt {
public:
    ReturnStmt(Expr *v, bool cc, SourcePos p);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

    Expr *val;
    /** This indicates whether the generated code will check to see if no
        more program instances are currently running after the return, in
        which case the code can possibly jump to the end of the current
        function. */
    const bool doCoherenceCheck;
};


/** Statement corresponding to a "case" label in the program.  In addition
    to the value associated with the "case", this statement also stores the
    statements following it. */
class CaseStmt : public Stmt {
public:
    CaseStmt(int value, Stmt *stmt, SourcePos pos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

    /** Integer value after the "case" statement */
    const int value;
    Stmt *stmts;
};


/** Statement for a "default" label (as would be found inside a "switch"
    statement). */
class DefaultStmt : public Stmt {
public:
    DefaultStmt(Stmt *stmt, SourcePos pos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

    Stmt *stmts;
};


/** A "switch" statement in the program. */
class SwitchStmt : public Stmt {
public:
    SwitchStmt(Expr *expr, Stmt *stmts, SourcePos pos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

    /** Expression that is used to determine which label to jump to. */
    Expr *expr;
    /** Statement block after the "switch" expression. */
    Stmt *stmts;
};


/** A "goto" in an ispc program. */
class GotoStmt : public Stmt {
public:
    GotoStmt(const char *label, SourcePos gotoPos, SourcePos idPos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *Optimize();
    Stmt *TypeCheck();
    int EstimateCost() const;

    /** Name of the label to jump to when the goto is executed. */
    std::string label;
    SourcePos identifierPos;
};


/** Statement corresponding to a label (as would be used as a goto target)
    in the program. */
class LabeledStmt : public Stmt {
public:
    LabeledStmt(const char *label, Stmt *stmt, SourcePos p);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *Optimize();
    Stmt *TypeCheck();
    int EstimateCost() const;

    /** Name of the label. */
    std::string name;
    /** Statements following the label. */
    Stmt *stmt;
};


/** @brief Representation of a list of statements in the program.
 */
class StmtList : public Stmt {
public:
    StmtList(SourcePos p) : Stmt(p) { }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

    void Add(Stmt *s) { if (s) stmts.push_back(s); }

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

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

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

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

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

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *TypeCheck();
    int EstimateCost() const;

    /** Expression that gives the pointer value to be deleted. */
    Expr *expr;
};

#endif // ISPC_STMT_H
