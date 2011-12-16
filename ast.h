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

/** @file ast.h
    @brief 
*/

#ifndef ISPC_AST_H
#define ISPC_AST_H 1

#include "ispc.h"
#include <vector>

/** @brief Abstract base class for nodes in the abstract syntax tree (AST).

    This class defines a basic interface that all abstract syntax tree
    (AST) nodes must implement.  The base classes for both expressions
    (Expr) and statements (Stmt) inherit from this class.
*/
class ASTNode {
public:
    ASTNode(SourcePos p) : pos(p) { }
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

    virtual int EstimateCost() const = 0;

    /** All AST nodes must track the file position where they are
        defined. */
    SourcePos pos;
};


/** Simple representation of the abstract syntax trees for all of the
    functions declared in a compilation unit.
 */
class AST {
public:
    /** Add the AST for a function described by the given declaration
        information and source code. */
    void AddFunction(Symbol *sym, const std::vector<Symbol *> &args, 
                     Stmt *code);

    /** Generate LLVM IR for all of the functions into the current
        module. */
    void GenerateIR();

private:
    std::vector<Function *> functions;
};


/** Callback function type for preorder traversial visiting function for
    the AST walk.
 */
typedef bool (* ASTPreCallBackFunc)(ASTNode *node, void *data);

/** Callback function type for postorder traversial visiting function for
    the AST walk.
 */
typedef ASTNode * (* ASTPostCallBackFunc)(ASTNode *node, void *data);

/** Walk (some portion of) an AST, starting from the given root node.  At
    each node, if preFunc is non-NULL, call it, passing the given void
    *data pointer; if the call to preFunc function returns false, then the
    children of the node aren't visited.  This function then makes
    recursive calls to WalkAST() to process the node's children; after
    doing so, calls postFunc, at the node.  The return value from the
    postFunc call is ignored. */
extern ASTNode *WalkAST(ASTNode *root, ASTPreCallBackFunc preFunc,
                        ASTPostCallBackFunc postFunc, void *data);

extern Expr *Optimize(Expr *);
extern Stmt *Optimize(Stmt *);
extern ASTNode *Optimize(ASTNode *root);

#endif // ISPC_AST_H
