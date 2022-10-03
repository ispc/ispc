/*
  Copyright (c) 2011-2022, Intel Corporation
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

/** @file ast.h
    @brief
*/

#pragma once

#include "ispc.h"
#include <string>
#include <vector>

namespace ispc {

/** @brief Helper class for printing AST.

    This class keeps track of indentation when printing AST.
    Before anything is printed, pushSingle() or pushList(int) methods need to be
    invoked to declare how the following node(s) are going to be printed - as
    a single nested node or a list of nested items. For example:

    Parent Node
    `-Single Nested Node

    Parent Node
    |-List Node #1
    |-List Node #2
    `-List Node #3

    If the nested node needs to be annotated, setNextLabel(string) should be called
    before recursing to the nested node.

    Parent Node
    |-(annotation 1) List Node #1
    |-(annotation 2) List Node #2
    `-(annotation 3) List Node #3

    The call to any of Print()/PrintLn() methods does the indentation. Every such
    call must be paired by Done() call when the node is printed.

    Note that this class is not assumed to encapsulate all printing functionality -
    Print() member function is responsible for printing indented beginning of the
    string, but the rest of the string needs to be printed with printf().
 */
class Indent {
    std::string label;
    std::vector<int> stack;
    int printCalls = 0;
    int doneCalls = 0;

  public:
    Indent() {}
    ~Indent();

    /** Declare that the next level of nesting will contain a single node. */
    void pushSingle();
    /** Declare that the next level of nesting will contain a list of nodes,
        where i is the number of expected nodes. If i is 0, then no nested nodes
        are expected (nothing is pushed on the stack).*/
    void pushList(int i);

    /** Annotate the next nested node with a label. */
    void setNextLabel(std::string s);

    /** Print indentation followed by an optional string string. */
    void Print(const char *title = nullptr);
    /** Print indentation followed by the string and source position. */
    void Print(const char *title, const SourcePos &pos);
    /** Print indentation followed by the string, source position and a new line character. */
    void PrintLn(const char *title, const SourcePos &pos);
    /** Declare that current node printing is done. */
    void Done();
};

/** @brief Abstract base class for nodes in the abstract syntax tree (AST).

    This class defines a basic interface that all abstract syntax tree
    (AST) nodes must implement.  The base classes for both expressions
    (Expr) and statements (Stmt) inherit from this class.
*/
class ASTNode {
    const unsigned char SubclassID; // Subclass identifier (for isa/dyn_cast)
  public:
    ASTNode(SourcePos p, unsigned scid) : SubclassID(scid), pos(p) {}
    virtual ~ASTNode();

    /** The Optimize() method should perform any appropriate early-stage
        optimizations on the node (e.g. constant folding).  This method
        will be called after the node's children have already been
        optimized, and the caller will store the returned ASTNode * in
        place of the original node.  This method should return NULL if an
        error is encountered during optimization. */
    virtual ASTNode *Optimize() = 0;

    /** Type checking should be performed by the node when this method is
        called.  In the event of an error, a NULL value may be returned.
        As with ASTNode::Optimize(), the caller should store the returned
        pointer in place of the original ASTNode *. */
    virtual ASTNode *TypeCheck() = 0;

    /** Estimate the execution cost of the node (not including the cost of
        the children.  The value returned should be based on the COST_*
        enumerant values defined in ispc.h. */
    virtual int EstimateCost() const = 0;

    /** All AST nodes must track the file position where they are
        defined. */
    SourcePos pos;

    /** An enumeration for keeping track of the concrete subclass of Value
        that is actually instantiated.*/
    enum ASTNodeTy {
        /* For classes inherited from Expr */
        AddressOfExprID,
        AllocaExprID,
        AssignExprID,
        BinaryExprID,
        ConstExprID,
        DerefExprID,
        PtrDerefExprID,
        RefDerefExprID,
        ExprListID,
        FunctionCallExprID,
        FunctionSymbolExprID,
        IndexExprID,
        StructMemberExprID,
        VectorMemberExprID,
        NewExprID,
        NullPointerExprID,
        ReferenceExprID,
        SelectExprID,
        SizeOfExprID,
        SymbolExprID,
        SyncExprID,
        TypeCastExprID,
        UnaryExprID,
        /* This is a convenience separator to shorten classof implementations */
        MaxExprID,
        /* For classes inherited from Stmt */
        AssertStmtID,
        BreakStmtID,
        CaseStmtID,
        ContinueStmtID,
        DeclStmtID,
        DefaultStmtID,
        DeleteStmtID,
        DoStmtID,
        ExprStmtID,
        ForeachActiveStmtID,
        ForeachStmtID,
        ForeachUniqueStmtID,
        ForStmtID,
        GotoStmtID,
        IfStmtID,
        LabeledStmtID,
        PrintStmtID,
        ReturnStmtID,
        StmtListID,
        SwitchStmtID,
        UnmaskedStmtID
    };

    /** Return an ID for the concrete type of this object. This is used to
        implement the classof checks.  This should not be used for any
        other purpose, as the values may change as ISPC evolves */
    unsigned getValueID() const { return SubclassID; }

    /** A function for interactive debugging */
    void Print() const;

    /** A function that should be used for hierarchical AST dump. */
    virtual void Print(Indent &indent) const = 0;

    static inline bool classof(ASTNode const *) { return true; }
};

class AST {
  public:
    /** Add the AST for a function described by the given declaration
        information and source code. */
    void AddFunction(Symbol *sym, Stmt *code);

    /** Generate LLVM IR for all of the functions into the current
        module. */
    void GenerateIR();

    void Print(Globals::ASTDumpKind printKind = Globals::ASTDumpKind::User) const;

  private:
    std::vector<Function *> functions;
};

/** Callback function type for preorder traversial visiting function for
    the AST walk.
 */
typedef bool (*ASTPreCallBackFunc)(ASTNode *node, void *data);

/** Callback function type for postorder traversial visiting function for
    the AST walk.
 */
typedef ASTNode *(*ASTPostCallBackFunc)(ASTNode *node, void *data);

/** Walk (some portion of) an AST, starting from the given root node.  At
    each node, if preFunc is non-NULL, call it, passing the given void
    *data pointer; if the call to preFunc function returns false, then the
    children of the node aren't visited.  This function then makes
    recursive calls to WalkAST() to process the node's children; after
    doing so, calls postFunc, at the node.  The return value from the
    postFunc call is ignored. */
extern ASTNode *WalkAST(ASTNode *root, ASTPreCallBackFunc preFunc, ASTPostCallBackFunc postFunc, void *data);

/** Perform simple optimizations on the AST or portion thereof passed to
    this function, returning the resulting AST. */
extern ASTNode *Optimize(ASTNode *root);

/** Convenience version of Optimize() for Expr *s that returns an Expr *
    (rather than an ASTNode *, which would require the caller to cast back
    to an Expr *). */
extern Expr *Optimize(Expr *);

/** Convenience version of Optimize() for Expr *s that returns an Stmt *
    (rather than an ASTNode *, which would require the caller to cast back
    to a Stmt *). */
extern Stmt *Optimize(Stmt *);

/** Perform type-checking on the given AST (or portion of one), returning a
    pointer to the root of the resulting AST. */
extern ASTNode *TypeCheck(ASTNode *root);

/** Convenience version of TypeCheck() for Expr *s that returns an Expr *. */
extern Expr *TypeCheck(Expr *);

/** Convenience version of TypeCheck() for Stmt *s that returns an Stmt *. */
extern Stmt *TypeCheck(Stmt *);

/** Returns an estimate of the execution cost of the tree starting at
    the given root. */
extern int EstimateCost(ASTNode *root);

/** Returns true if it would be safe to run the given code with an "all
    off" mask. */
extern bool SafeToRunWithMaskAllOff(ASTNode *root);

} // namespace ispc
