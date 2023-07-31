/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
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
    which in turn records the per-variable information like the name, array
    size (if any), initializer expression, etc.
*/

#pragma once

#include "ast.h"
#include "ispc.h"

#include <llvm/ADT/SmallVector.h>

namespace ispc {

struct VariableDeclaration;

class Declaration;
class Declarator;

/* Multiple qualifiers can be provided with types in declarations;
   therefore, they are set up so that they can be ANDed together into an
   int. */
#define TYPEQUAL_NONE 0
#define TYPEQUAL_CONST (1 << 0)
#define TYPEQUAL_UNIFORM (1 << 1)
#define TYPEQUAL_VARYING (1 << 2)
#define TYPEQUAL_TASK (1 << 3)
#define TYPEQUAL_SIGNED (1 << 4)
#define TYPEQUAL_UNSIGNED (1 << 5)
#define TYPEQUAL_INLINE (1 << 6)
#define TYPEQUAL_EXPORT (1 << 7)
#define TYPEQUAL_UNMASKED (1 << 8)
#define TYPEQUAL_NOINLINE (1 << 9)
#define TYPEQUAL_VECTORCALL (1 << 10)
#define TYPEQUAL_REGCALL (1 << 11)

/** @brief Representation of the declaration specifiers in a declaration.

    In other words, this represents all of the stuff that applies to all of
    the (possibly multiple) variables in a declaration.
 */
class DeclSpecs : public Traceable {
  public:
    DeclSpecs(const Type *t = nullptr, StorageClass sc = SC_NONE, int tq = TYPEQUAL_NONE);

    void Print() const;

    StorageClass storageClass;

    /** Zero or more of the TYPEQUAL_* values, ANDed together. */
    int typeQualifiers;

    /** The basic type provided in the declaration; this should be an
        AtomicType, EnumType, StructType, or VectorType; other types (like
        ArrayTypes) will end up being created if a particular declaration
        has an array size, etc.
    */
    const Type *baseType;

    const Type *GetBaseType(SourcePos pos) const;

    /** If this is a declaration with a vector type, this gives the vector
        width.  For non-vector types, this is zero.
     */
    int vectorSize;

    /** If this is a declaration with an "soa<n>" qualifier, this gives the
        SOA width specified.  Otherwise this is zero.
     */
    int soaWidth;

    std::vector<std::pair<std::string, SourcePos>> declSpecList;
};

enum DeclaratorKind { DK_BASE, DK_POINTER, DK_REFERENCE, DK_ARRAY, DK_FUNCTION };

/** @brief Representation of the declaration of a single variable.

    In conjunction with an instance of the DeclSpecs, this gives us
    everything we need for a full variable declaration.
 */
class Declarator : public Traceable {
  public:
    Declarator(DeclaratorKind dk, SourcePos p);

    /** Once a DeclSpecs instance is available, this method completes the
        initialization of the type member.
     */
    void InitFromDeclSpecs(DeclSpecs *ds);

    void InitFromType(const Type *base, DeclSpecs *ds);

    void Print() const;
    void Print(Indent &indent) const;

    /** Position of the declarator in the source program. */
    const SourcePos pos;

    /** The kind of this declarator; complex declarations are assembled as
        a hierarchy of Declarators.  (For example, a pointer to an int
        would have a root declarator with kind DK_POINTER and with the
        Declarator::child member pointing to a DK_BASE declarator for the
        int). */
    const DeclaratorKind kind;

    /** Child pointer if needed; this can only be non-nullptr if the
        declarator's kind isn't DK_BASE. */
    Declarator *child;

    /** Type qualifiers provided with the declarator. */
    int typeQualifiers;

    StorageClass storageClass;

    /** For array declarators, this gives the declared size of the array.
        Unsized arrays have arraySize == 0. */
    int arraySize;

    /** Name associated with the declarator. */
    std::string name;

    /** Initialization expression for the variable.  May be nullptr. */
    Expr *initExpr;

    /** Type of the declarator.  This is nullptr until InitFromDeclSpecs() or
        InitFromType() is called. */
    const Type *type;

    /** For function declarations, this holds the Declaration *s for the
        function's parameters. */
    std::vector<Declaration *> functionParams;
};

/** @brief Representation of a full declaration of one or more variables,
    including the shared DeclSpecs as well as the per-variable Declarators.
 */
class Declaration : public Traceable {
  public:
    Declaration(DeclSpecs *ds, std::vector<Declarator *> *dlist = nullptr);
    Declaration(DeclSpecs *ds, Declarator *d);

    void Print() const;
    void Print(Indent &indent) const;

    /** This method walks through all of the Declarators in a declaration
        and returns a fully-initialized Symbol and (possibly) and
        initialization expression for each one.  (This allows the rest of
        the system to not have to worry about the mess of the general
        Declarator representation.) */
    std::vector<VariableDeclaration> GetVariableDeclarations() const;

    /** For any function declarations in the Declaration, add the
        declaration to the module. */
    void DeclareFunctions();

    DeclSpecs *declSpecs;
    std::vector<Declarator *> declarators;
};

/** The parser creates instances of StructDeclaration for the members of
    structs as it's parsing their declarations. */
struct StructDeclaration : public Traceable {
    StructDeclaration(const Type *t, std::vector<Declarator *> *d) : type(t), declarators(d) {}
    ~StructDeclaration() { delete declarators; }

    // We don't copy these objects at the moment. If we will then proper
    // implementations are needed considering the ownership of declarators.
    StructDeclaration(const StructDeclaration &) = delete;
    StructDeclaration &operator=(const StructDeclaration &) = delete;

    const Type *type;
    std::vector<Declarator *> *declarators;
};

/** Given a set of StructDeclaration instances, this returns the types of
    the elements of the corresponding struct and their names. */
extern void GetStructTypesNamesPositions(const std::vector<StructDeclaration *> &sd,
                                         llvm::SmallVector<const Type *, 8> *elementTypes,
                                         llvm::SmallVector<std::string, 8> *elementNames,
                                         llvm::SmallVector<SourcePos, 8> *elementPositions);
} // namespace ispc
