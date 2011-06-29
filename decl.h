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

/** @file decl.h
    @brief Declarations related to type declarations; the parser basically
    creates instances of these classes, which are then turned into actual
    Types.

    Three classes work together to represent declarations.  As an example,
    consider a declaration like:

    static uniform int foo, bar[10];

    An instance of the Declaration class represents this entire declaration
    of two variables, 'foo' and 'bar'.  It holds a single instance of the
    DeclSpecs class represents the common specifiers for all of the
    variables--here, that the declaration has the 'static' and 'uniform'
    qualifiers, and that it's basic type is 'int'.  Then for each variable
    declaration, the Declaraiton class holds an instance of a Declarator,
    which in turn records the per-variable information like the symbol
    name, array size (if any), initializer expression, etc.
*/

#ifndef ISPC_DECL_H
#define ISPC_DECL_H

#include "ispc.h"

enum StorageClass {
    SC_NONE,
    SC_EXTERN,
    SC_EXPORT,
    SC_STATIC,
    SC_TYPEDEF,
    SC_EXTERN_C
};


/* Multiple qualifiers can be provided with types in declarations;
   therefore, they are set up so that they can be ANDed together into an
   int. */
#define TYPEQUAL_NONE           0
#define TYPEQUAL_CONST      (1<<0)
#define TYPEQUAL_UNIFORM    (1<<1)
#define TYPEQUAL_VARYING    (1<<2)
#define TYPEQUAL_TASK       (1<<3)
#define TYPEQUAL_REFERENCE  (1<<4)
#define TYPEQUAL_UNSIGNED   (1<<5)
#define TYPEQUAL_INLINE     (1<<6)

/** @brief Representation of the declaration specifiers in a declaration.

    In other words, this represents all of the stuff that applies to all of
    the (possibly multiple) variables in a declaration.
 */
class DeclSpecs {
public:
    DeclSpecs(const Type *t = NULL, StorageClass sc = SC_NONE, int tq = TYPEQUAL_NONE);

    void Print() const;

    StorageClass storageClass;

    /** Zero or more of the TYPEQUAL_* values, ANDed together. */
    int typeQualifier;

    /** The basic type provided in the declaration; this should be an
        AtomicType, a StructType, or a VectorType; other types (like
        ArrayTypes) will end up being created if a particular declaration
        has an array size, etc.
    */
    const Type *baseType;

    /** If this is a declaration with a vector type, this gives the vector
        width.  For non-vector types, this is zero.
     */
    int vectorSize;

    /** If this is a declaration with an "soa<n>" qualifier, this gives the
        SOA width specified.  Otherwise this is zero.
     */
    int soaWidth;
};


/** @brief Representation of the declaration of a single variable.  

    In conjunction with an instance of the DeclSpecs, this gives us
    everything we need for a full variable declaration.
 */
class Declarator {
public:
    Declarator(Symbol *s, SourcePos p);

    /** As the parser peels off array dimension declarations after the
        symbol name, it calls this method to provide them to the
        Declarator.
     */
    void AddArrayDimension(int size);

    /** Once a DeclSpecs instance is available, this method completes the
        initialization of the Symbol, setting its Type accordingly.
     */
    void InitFromDeclSpecs(DeclSpecs *ds);

    /** Get the actual type of the combination of Declarator and the given
        DeclSpecs */
    const Type *GetType(DeclSpecs *ds) const;

    void Print() const;

    const SourcePos pos;
    Symbol *sym;
    /** If this declarator includes an array specification, the sizes of
        the array dimensions are represented here.
     */
    std::vector<int> arraySize;
    /** Initialization expression for the variable.  May be NULL. */
    Expr *initExpr;
    bool isFunction;
    std::vector<Declaration *> *functionArgs;
};


/** @brief Representation of a full declaration of one or more variables,
    including the shared DeclSpecs as well as the per-variable Declarators.
 */
class Declaration {
public:
    Declaration(DeclSpecs *ds, std::vector<Declarator *> *dlist = NULL) {
        declSpecs = ds;
        if (dlist != NULL)
            declarators = *dlist;
        for (unsigned int i = 0; i < declarators.size(); ++i)
            if (declarators[i] != NULL)
                declarators[i]->InitFromDeclSpecs(declSpecs);
    }
    Declaration(DeclSpecs *ds, Declarator *d) {
        declSpecs = ds;
        if (d) {
            d->InitFromDeclSpecs(ds);
            declarators.push_back(d);
        }
    }

    /** Adds the symbols for the variables in the declaration to the symbol
        table. */
    void AddSymbols(SymbolTable *st) const;
    void Print() const;

    DeclSpecs *declSpecs;
    std::vector<Declarator *> declarators;
};


/** The parser creates instances of StructDeclaration for the members of
    structs as it's parsing their declarations. */
struct StructDeclaration {
    StructDeclaration(const Type *t, std::vector<Declarator *> *d)
        : type(t), declarators(d) { }

    const Type *type;
    std::vector<Declarator *> *declarators;
};


/** Given a set of StructDeclaration instances, this returns the types of
    the elements of the corresponding struct and their names. */
extern void GetStructTypesNamesPositions(const std::vector<StructDeclaration *> &sd,
                                         std::vector<const Type *> *elementTypes,
                                         std::vector<std::string> *elementNames,
                                         std::vector<SourcePos> *elementPositions);

#endif // ISPC_DECL_H
