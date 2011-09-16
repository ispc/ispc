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
    // ASTNode *s, as in the original ASTNode declarations of them.
    virtual Stmt *Optimize() = 0;
    virtual Stmt *TypeCheck() = 0;
};


/** @brief Statement representing a single expression */
class ExprStmt : public Stmt {
public:
    ExprStmt(Expr *expr, SourcePos pos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *Optimize();
    Stmt *TypeCheck();

private:
    Expr *expr;
};


/** @brief Statement representing a single declaration (which in turn may declare
    a number of variables. */
class DeclStmt : public Stmt {
public:
    DeclStmt(SourcePos pos, Declaration *declaration, SymbolTable *symbolTable);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *Optimize();
    Stmt *TypeCheck();

private:
    Declaration *declaration;
};


/** @brief Statement representing a single if statement, possibly with an
    else clause. */
class IfStmt : public Stmt {
public:
    IfStmt(Expr *testExpr, Stmt *trueStmts, Stmt *falseStmts,
           bool doAllCheck, SourcePos pos);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *Optimize();
    Stmt *TypeCheck();

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
    const bool doAnyCheck;

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

    Stmt *Optimize();
    Stmt *TypeCheck();

private:
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

    Stmt *Optimize();
    Stmt *TypeCheck();

private:
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

    Stmt *Optimize();
    Stmt *TypeCheck();

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

    Stmt *Optimize();
    Stmt *TypeCheck();

private:
    /** This indicates whether the generated code will check to see if no
        more program instances are currently running after the continue, in
        which case the code can have a jump to the end of the current
        loop. */
    const bool doCoherenceCheck;
};


/** @brief Statement implementation for a 'return' or 'coherent' return
    statement in the program. */
class ReturnStmt : public Stmt {
public:
    ReturnStmt(Expr *v, bool cc, SourcePos p);

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *Optimize();
    Stmt *TypeCheck();

private:
    Expr *val;
    /** This indicates whether the generated code will check to see if no
        more program instances are currently running after the return, in
        which case the code can possibly jump to the end of the current
        function. */
    const bool doCoherenceCheck;
};


/** @brief Representation of a list of statements in the program.
 */
class StmtList : public Stmt {
public:
    StmtList(SourcePos p) : Stmt(p) { }

    void EmitCode(FunctionEmitContext *ctx) const;
    void Print(int indent) const;

    Stmt *Optimize();
    Stmt *TypeCheck();

    void Add(Stmt *s) { if (s) stmts.push_back(s); }
    const std::vector<Stmt *> &GetStatements() { return stmts; }

private:
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

    Stmt *Optimize();
    Stmt *TypeCheck();

private:
    /** Format string for the print() statement. */
    const std::string format;
    /** This holds the arguments passed to the print() statement.  If more
        than one was provided, this will be an ExprList. */
    Expr *values;
};


#endif // ISPC_STMT_H
