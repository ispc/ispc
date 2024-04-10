/*
  Copyright (c) 2010-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file expr.h
    @brief Expr abstract base class and expression implementations
*/

#pragma once

#include "ast.h"
#include "ctx.h"
#include "func.h"
#include "ispc.h"
#include "type.h"

namespace ispc {

/** @brief Expr is the abstract base class that defines the interface that
    all expression types must implement.
 */
class Expr : public ASTNode {
  public:
    Expr(SourcePos p, unsigned scid) : ASTNode(p, scid) {}

    static inline bool classof(Expr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() < MaxExprID; }

    /** This is the main method for Expr implementations to implement.  It
        should call methods in the FunctionEmitContext to emit LLVM IR
        instructions to the current basic block in order to generate an
        llvm::Value that represents the expression's value. */
    virtual llvm::Value *GetValue(FunctionEmitContext *ctx) const = 0;

    /** For expressions that can provide an lvalue (e.g. array indexing),
        this function should emit IR that computes the expression's lvalue
        and returns the corresponding llvm::Value.  Expressions that can't
        provide an lvalue should leave this unimplemented; the default
        implementation returns nullptr.  */
    virtual llvm::Value *GetLValue(FunctionEmitContext *ctx) const;

    /** Returns the Type of the expression. */
    virtual const Type *GetType() const = 0;

    /** Returns the type of the value returned by GetLValueType(); this
        should be a pointer type of some sort (uniform or varying). */
    virtual const Type *GetLValueType() const;

    /** For expressions that have values based on a symbol (e.g. regular
        symbol references, array indexing, etc.), this returns a pointer to
        that symbol. */
    virtual Symbol *GetBaseSymbol() const;

    /** If this is a constant expression that can be converted to a
        constant of storage type of the given type, this method should return the
        corresponding llvm::Constant value and a flag denoting if it's
        valid for multi-target compilation for use as an initializer of
        a global variable. Otherwise it should return the llvm::constant
        value as nullptr. */
    virtual std::pair<llvm::Constant *, bool> GetStorageConstant(const Type *type) const;

    /** If this is a constant expression that can be converted to a
        constant of the given type, this method should return the
        corresponding llvm::Constant value and a flag denoting if it's
        valid for multi-target compilation for use as an initializer of
        a global variable. Otherwise it should return the llvm::constant
        value as nullptr. */
    virtual std::pair<llvm::Constant *, bool> GetConstant(const Type *type) const;

    /** This method should perform early optimizations of the expression
        (constant folding, etc.) and return a pointer to the resulting
        expression.  If an error is encountered during optimization, nullptr
        should be returned. */
    virtual Expr *Optimize() = 0;

    /** This method should perform type checking of the expression and
        return a pointer to the resulting expression.  If an error is
        encountered, nullptr should be returned. */
    virtual Expr *TypeCheck() = 0;

    virtual Expr *Instantiate(TemplateInstantiation &templInst) const = 0;

    virtual bool HasAmbiguousVariability(std::vector<const Expr *> &warn) const;
};

/** @brief Unary expression */
class UnaryExpr : public Expr {
  public:
    enum Op {
        PreInc,     ///< Pre-increment
        PreDec,     ///< Pre-decrement
        PostInc,    ///< Post-increment
        PostDec,    ///< Post-decrement
        Negate,     ///< Negation
        LogicalNot, ///< Logical not
        BitNot,     ///< Bit not
    };

    UnaryExpr(Op op, Expr *expr, SourcePos pos);

    static inline bool classof(UnaryExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == UnaryExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    void Print(Indent &indent) const;
    Expr *Optimize();
    Expr *TypeCheck();
    int EstimateCost() const;
    UnaryExpr *Instantiate(TemplateInstantiation &templInst) const;

    const Op op;
    Expr *expr;
};

/** @brief Binary expression */
class BinaryExpr : public Expr {
  public:
    enum Op {
        Add, ///< Addition
        Sub, ///< Subtraction
        Mul, ///< Multiplication
        Div, ///< Division
        Mod, ///< Modulus
        Shl, ///< Shift left
        Shr, ///< Shift right

        Lt,       ///< Less than
        Gt,       ///< Greater than
        Le,       ///< Less than or equal
        Ge,       ///< Greater than or equal
        Equal,    ///< Equal
        NotEqual, ///< Not equal

        BitAnd,     ///< Bitwise AND
        BitXor,     ///< Bitwise XOR
        BitOr,      ///< Bitwise OR
        LogicalAnd, ///< Logical AND
        LogicalOr,  ///< Logical OR

        Comma, ///< Comma operator
    };

    BinaryExpr(Op o, Expr *a, Expr *b, SourcePos p);

    static inline bool classof(BinaryExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == BinaryExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    const Type *GetLValueType() const;
    void Print(Indent &indent) const;

    Expr *Optimize();
    Expr *TypeCheck();
    int EstimateCost() const;
    Expr *Instantiate(TemplateInstantiation &templInst) const;
    std::pair<llvm::Constant *, bool> GetStorageConstant(const Type *type) const;
    std::pair<llvm::Constant *, bool> GetConstant(const Type *type) const;
    bool HasAmbiguousVariability(std::vector<const Expr *> &warn) const;

    const Op op;
    Expr *arg0, *arg1;
};

/** @brief Assignment expression */
class AssignExpr : public Expr {
  public:
    enum Op {
        Assign,    ///< Regular assignment
        MulAssign, ///< *= assignment
        DivAssign, ///< /= assignment
        ModAssign, ///< %= assignment
        AddAssign, ///< += assignment
        SubAssign, ///< -= assignment
        ShlAssign, ///< <<= assignment
        ShrAssign, ///< >>= assignment
        AndAssign, ///< &= assignment
        XorAssign, ///< ^= assignment
        OrAssign,  ///< |= assignment
    };

    AssignExpr(Op o, Expr *a, Expr *b, SourcePos p);

    static inline bool classof(AssignExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == AssignExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    void Print(Indent &indent) const;

    Expr *Optimize();
    Expr *TypeCheck();
    int EstimateCost() const;
    AssignExpr *Instantiate(TemplateInstantiation &templInst) const;

    const Op op;
    Expr *lvalue, *rvalue;
};

/** @brief Selection expression, corresponding to "test ? a : b".

    Returns the value of "a" or "b", depending on the value of "test".
*/
class SelectExpr : public Expr {
  public:
    SelectExpr(Expr *test, Expr *a, Expr *b, SourcePos p);

    static inline bool classof(SelectExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == SelectExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    const Type *GetLValueType() const;
    void Print(Indent &indent) const;

    Expr *Optimize();
    Expr *TypeCheck();
    int EstimateCost() const;
    SelectExpr *Instantiate(TemplateInstantiation &templInst) const;
    bool HasAmbiguousVariability(std::vector<const Expr *> &warn) const;

    Expr *test, *expr1, *expr2;
};

/** @brief A list of expressions.

    These are mostly used for representing curly-brace delimited
    initializers for initializers for complex types and for representing
    the arguments passed to a function call.
 */
class ExprList : public Expr {
  public:
    ExprList(SourcePos p) : Expr(p, ExprListID) {}
    ExprList(Expr *e, SourcePos p) : Expr(p, ExprListID) { exprs.push_back(e); }

    static inline bool classof(ExprList const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == ExprListID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    void Print(Indent &indent) const;
    std::pair<llvm::Constant *, bool> GetStorageConstant(const Type *type) const;
    std::pair<llvm::Constant *, bool> GetConstant(const Type *type) const;
    ExprList *Optimize();
    ExprList *TypeCheck();
    int EstimateCost() const;
    ExprList *Instantiate(TemplateInstantiation &templInst) const;
    bool HasAmbiguousVariability(std::vector<const Expr *> &warn) const;

    std::vector<Expr *> exprs;

    // Utility structs to support initializers lists for vectors
    struct ExprPosMapping {
        Expr *expr;
        int pos;
        ExprPosMapping(Expr *e, int p) : expr(e), pos(p) {}
    };

    struct ExprPosMappingVectorForVector {
        int vec_mem_pos_from;
        int vec_mem_pos_to;
        Expr *expr;
        ExprPosMappingVectorForVector(int from, int to, Expr *e)
            : vec_mem_pos_from(from), vec_mem_pos_to(to), expr(e) {}
        ExprPosMappingVectorForVector(int from, ExprPosMapping map)
            : vec_mem_pos_from(from), vec_mem_pos_to(map.pos), expr(map.expr) {}
    };

    // Returns true if each expression in expression list has AtomicType.
    // It also constructs a map of initializers for each atomic basetype.
    bool HasAtomicInitializerList(std::map<AtomicType::BasicType, std::vector<ExprPosMapping>> &map);
};

/** @brief Expression representing a function call.
 */
class FunctionCallExpr : public Expr {
  public:
    FunctionCallExpr(Expr *func, ExprList *args, SourcePos p, bool isLaunch = false, Expr *launchCountExpr[3] = nullptr,
                     bool isInvoke = false);

    static inline bool classof(FunctionCallExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == FunctionCallExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    llvm::Value *GetLValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    const Type *GetLValueType() const;
    void Print(Indent &indent) const;

    Expr *Optimize();
    Expr *TypeCheck();
    int EstimateCost() const;
    FunctionCallExpr *Instantiate(TemplateInstantiation &templInst) const;

    Expr *func;
    ExprList *args;
    bool isLaunch;
    bool isInvoke;
    Expr *launchCountExpr[3];
};

/** @brief Expression representing indexing into something with an integer
    offset.

    This is used for both array indexing and indexing into VectorTypes.
*/
class IndexExpr : public Expr {
  public:
    IndexExpr(Expr *baseExpr, Expr *index, SourcePos p);

    static inline bool classof(IndexExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == IndexExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    llvm::Value *GetLValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    const Type *GetLValueType() const;
    Symbol *GetBaseSymbol() const;
    void Print(Indent &indent) const;

    Expr *Optimize();
    Expr *TypeCheck();
    int EstimateCost() const;
    IndexExpr *Instantiate(TemplateInstantiation &templInst) const;

    Expr *baseExpr, *index;

  private:
    mutable const Type *type;
    mutable const PointerType *lvalueType;
};

/** @brief Expression representing member selection ("foo.bar").
 *
 *  This will also be overloaded to deal with swizzles.
 */
class MemberExpr : public Expr {
  public:
    static MemberExpr *create(Expr *expr, const char *identifier, SourcePos pos, SourcePos identifierPos,
                              bool derefLvalue);

    static inline bool classof(MemberExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) {
        return ((N->getValueID() == StructMemberExprID) || (N->getValueID() == VectorMemberExprID) ||
                (N->getValueID() == DependentMemberExprID));
    }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    llvm::Value *GetLValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    Symbol *GetBaseSymbol() const;
    void Print(Indent &indent) const;
    Expr *Optimize();
    Expr *TypeCheck();
    int EstimateCost() const;
    MemberExpr *Instantiate(TemplateInstantiation &templInst) const;

    virtual int getElementNumber() const = 0;
    virtual const Type *getElementType() const = 0;
    std::string getCandidateNearMatches() const;

    Expr *expr;
    std::string identifier;
    const SourcePos identifierPos;

    MemberExpr(Expr *expr, const char *identifier, SourcePos pos, SourcePos identifierPos, bool derefLValue,
               unsigned scid);

    /** Indicates whether the expression should be dereferenced before the
        member is found.  (i.e. this is true if the MemberExpr was a '->'
        operator, and is false if it was a '.' operator. */
    bool dereferenceExpr;

  protected:
    mutable const Type *type, *lvalueType;
};

class StructMemberExpr : public MemberExpr {
  public:
    StructMemberExpr(Expr *e, const char *id, SourcePos p, SourcePos idpos, bool derefLValue);

    static inline bool classof(StructMemberExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == StructMemberExprID; }

    const Type *GetType() const;
    const Type *GetLValueType() const;
    int getElementNumber() const;
    const Type *getElementType() const;

  private:
    const StructType *getStructType() const;
};

class VectorMemberExpr : public MemberExpr {
  public:
    VectorMemberExpr(Expr *e, const char *id, SourcePos p, SourcePos idpos, bool derefLValue);

    static inline bool classof(VectorMemberExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == VectorMemberExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    llvm::Value *GetLValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    const Type *GetLValueType() const;

    int getElementNumber() const;
    const Type *getElementType() const;

  private:
    const VectorType *exprVectorType;
    const VectorType *memberType;
};

class DependentMemberExpr : public MemberExpr {
  public:
    DependentMemberExpr(Expr *e, const char *id, SourcePos p, SourcePos idpos, bool derefLValue);

    static inline bool classof(DependentMemberExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == DependentMemberExprID; }

    int getElementNumber() const;
    const Type *getElementType() const;
};

/** @brief Expression representing a compile-time constant value.

    This class can currently represent compile-time constants of anything
    that is an AtomicType or an EnumType; for anything more complex, we
    don't currently have a representation of a compile-time constant that
    can be further reasoned about.
 */
class ConstExpr : public Expr {
  public:
    /** Create a ConstExpr from a uniform int8 value */
    ConstExpr(const Type *t, int8_t i, SourcePos p);
    /** Create a ConstExpr from a varying int8 value */
    ConstExpr(const Type *t, int8_t *i, SourcePos p);
    /** Create a ConstExpr from a uniform uint8 value */
    ConstExpr(const Type *t, uint8_t u, SourcePos p);
    /** Create a ConstExpr from a varying uint8 value */
    ConstExpr(const Type *t, uint8_t *u, SourcePos p);

    /** Create a ConstExpr from a uniform int16 value */
    ConstExpr(const Type *t, int16_t i, SourcePos p);
    /** Create a ConstExpr from a varying int16 value */
    ConstExpr(const Type *t, int16_t *i, SourcePos p);
    /** Create a ConstExpr from a uniform uint16 value */
    ConstExpr(const Type *t, uint16_t u, SourcePos p);
    /** Create a ConstExpr from a varying uint16 value */
    ConstExpr(const Type *t, uint16_t *u, SourcePos p);

    /** Create a ConstExpr from a uniform int32 value */
    ConstExpr(const Type *t, int32_t i, SourcePos p);
    /** Create a ConstExpr from a varying int32 value */
    ConstExpr(const Type *t, int32_t *i, SourcePos p);
    /** Create a ConstExpr from a uniform uint32 value */
    ConstExpr(const Type *t, uint32_t u, SourcePos p);
    /** Create a ConstExpr from a varying uint32 value */
    ConstExpr(const Type *t, uint32_t *u, SourcePos p);

    /** Create a ConstExpr from a llvm::APFloat value for uniform floating point types */
    ConstExpr(const Type *t, llvm::APFloat f, SourcePos p);
    /** Create a ConstExpr from a llvm::APFloat value for varying floating point types */
    ConstExpr(const Type *t, std::vector<llvm::APFloat> const &f, SourcePos p);

    /** Create a ConstExpr from a uniform int64 value */
    ConstExpr(const Type *t, int64_t i, SourcePos p);
    /** Create a ConstExpr from a varying int64 value */
    ConstExpr(const Type *t, int64_t *i, SourcePos p);
    /** Create a ConstExpr from a uniform uint64 value */
    ConstExpr(const Type *t, uint64_t i, SourcePos p);
    /** Create a ConstExpr from a varying uint64 value */
    ConstExpr(const Type *t, uint64_t *i, SourcePos p);

    /** Create a ConstExpr from a uniform bool value */
    ConstExpr(const Type *t, bool b, SourcePos p);
    /** Create a ConstExpr from a varying bool value */
    ConstExpr(const Type *t, bool *b, SourcePos p);

    /** Create ConstExpr with the same type and values as the given one,
        but at the given position. */
    ConstExpr(const ConstExpr *old, SourcePos pos);

    static inline bool classof(ConstExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == ConstExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    void Print(Indent &indent) const;
    std::pair<llvm::Constant *, bool> GetStorageConstant(const Type *type) const;
    std::pair<llvm::Constant *, bool> GetConstant(const Type *constType) const;

    Expr *TypeCheck();
    Expr *Optimize();
    int EstimateCost() const;
    ConstExpr *Instantiate(TemplateInstantiation &templInst) const;

    /** Return the ConstExpr's values as the given pointer type, doing type
        conversion from the actual type if needed.  If forceVarying is
        true, then type convert to 'varying' so as to always return a
        number of values equal to the target vector width into the given
        pointer. */
    int GetValues(bool *, bool forceVarying = false) const;
    int GetValues(int8_t *, bool forceVarying = false) const;
    int GetValues(uint8_t *, bool forceVarying = false) const;
    int GetValues(int16_t *, bool forceVarying = false) const;
    int GetValues(uint16_t *, bool forceVarying = false) const;
    int GetValues(std::vector<llvm::APFloat> &, llvm::Type *, bool forceVarying = false) const;
    int GetValues(int32_t *, bool forceVarying = false) const;
    int GetValues(uint32_t *, bool forceVarying = false) const;
    int GetValues(int64_t *, bool forceVarying = false) const;
    int GetValues(uint64_t *, bool forceVarying = false) const;
    int GetValues(std::vector<llvm::APFloat> &) const;

    /** Return the ConstExpr's values as a string. */
    std::string GetValuesAsStr(const std::string &separator) const;

    /** Return the number of values in the ConstExpr; should be either 1,
        if it has uniform type, or the target's vector width if it's
        varying. */
    int Count() const;

    /** Return true if the type and values of two ConstExpr are the same. */
    bool IsEqual(const ConstExpr *ce) const;

  private:
    AtomicType::BasicType getBasicType() const;

    const Type *type;
    union {
        int8_t int8Val[ISPC_MAX_NVEC];
        uint8_t uint8Val[ISPC_MAX_NVEC];
        int16_t int16Val[ISPC_MAX_NVEC];
        uint16_t uint16Val[ISPC_MAX_NVEC];
        int32_t int32Val[ISPC_MAX_NVEC];
        uint32_t uint32Val[ISPC_MAX_NVEC];
        bool boolVal[ISPC_MAX_NVEC];
        int64_t int64Val[ISPC_MAX_NVEC];
        uint64_t uint64Val[ISPC_MAX_NVEC];
    };
    std::vector<llvm::APFloat> fpVal;
};

/** @brief Expression representing a type cast of the given expression to a
    probably-different type. */
class TypeCastExpr : public Expr {
  public:
    TypeCastExpr(const Type *t, Expr *e, SourcePos p);

    static inline bool classof(TypeCastExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == TypeCastExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    llvm::Value *GetLValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    const Type *GetLValueType() const;
    void Print(Indent &indent) const;
    Expr *TypeCheck();
    Expr *Optimize();
    int EstimateCost() const;
    TypeCastExpr *Instantiate(TemplateInstantiation &templInst) const;
    Symbol *GetBaseSymbol() const;
    std::pair<llvm::Constant *, bool> GetConstant(const Type *type) const;
    bool HasAmbiguousVariability(std::vector<const Expr *> &warn) const;
    void PrintAmbiguousVariability() const;

    const Type *type;
    Expr *expr;
};

/** @brief Expression that represents taking a reference of a (non-reference)
    variable. */
class ReferenceExpr : public Expr {
  public:
    ReferenceExpr(Expr *e, SourcePos p);

    static inline bool classof(ReferenceExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == ReferenceExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    const Type *GetLValueType() const;
    Symbol *GetBaseSymbol() const;
    void Print(Indent &indent) const;
    Expr *TypeCheck();
    Expr *Optimize();
    int EstimateCost() const;
    ReferenceExpr *Instantiate(TemplateInstantiation &templInst) const;

    Expr *expr;
};

/** @brief Common base class that provides shared functionality for
    PtrDerefExpr and RefDerefExpr. */
class DerefExpr : public Expr {
  public:
    DerefExpr(Expr *e, SourcePos p, unsigned scid = DerefExprID);

    static inline bool classof(DerefExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) {
        return ((N->getValueID() == DerefExprID) || (N->getValueID() == PtrDerefExprID) ||
                (N->getValueID() == RefDerefExprID));
    }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    llvm::Value *GetLValue(FunctionEmitContext *ctx) const;
    const Type *GetLValueType() const;
    Symbol *GetBaseSymbol() const;
    Expr *Optimize();

    Expr *expr;
};

/** @brief Expression that represents dereferencing a pointer to get its
    value. */
class PtrDerefExpr : public DerefExpr {
  public:
    PtrDerefExpr(Expr *e, SourcePos p);

    static inline bool classof(PtrDerefExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == PtrDerefExprID; }

    const Type *GetType() const;
    void Print(Indent &indent) const;
    Expr *TypeCheck();
    int EstimateCost() const;
    PtrDerefExpr *Instantiate(TemplateInstantiation &templInst) const;
};

/** @brief Expression that represents dereferencing a reference to get its
    value. */
class RefDerefExpr : public DerefExpr {
  public:
    RefDerefExpr(Expr *e, SourcePos p);

    static inline bool classof(RefDerefExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == RefDerefExprID; }

    const Type *GetType() const;
    void Print(Indent &indent) const;
    Expr *TypeCheck();
    int EstimateCost() const;
    RefDerefExpr *Instantiate(TemplateInstantiation &templInst) const;
};

/** Expression that represents taking the address of an expression. */
class AddressOfExpr : public Expr {
  public:
    AddressOfExpr(Expr *e, SourcePos p);

    static inline bool classof(AddressOfExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == AddressOfExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    const Type *GetLValueType() const;
    Symbol *GetBaseSymbol() const;
    void Print(Indent &indent) const;
    Expr *TypeCheck();
    Expr *Optimize();
    int EstimateCost() const;
    AddressOfExpr *Instantiate(TemplateInstantiation &templInst) const;
    std::pair<llvm::Constant *, bool> GetConstant(const Type *type) const;

    Expr *expr;
};

/** Expression that returns the size of the given expression or type in
    bytes. */
class SizeOfExpr : public Expr {
  public:
    SizeOfExpr(Expr *e, SourcePos p);
    SizeOfExpr(const Type *t, SourcePos p);

    static inline bool classof(SizeOfExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == SizeOfExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    void Print(Indent &indent) const;
    Expr *TypeCheck();
    Expr *Optimize();
    int EstimateCost() const;
    SizeOfExpr *Instantiate(TemplateInstantiation &templInst) const;
    std::pair<llvm::Constant *, bool> GetConstant(const Type *type) const;

    /* One of expr or type should be non-nullptr (but not both of them).  The
       SizeOfExpr returns the size of whichever one of them isn't nullptr. */
    Expr *expr;
    const Type *type;
};

//  Expression that allocates space in the stack frame of the caller
//  and returns a pointer to the beginning of the allocated space.
class AllocaExpr : public Expr {
  public:
    AllocaExpr(Expr *e, SourcePos p);

    static inline bool classof(AllocaExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == AllocaExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    void Print(Indent &indent) const;
    Expr *TypeCheck();
    Expr *Optimize();
    int EstimateCost() const;
    AllocaExpr *Instantiate(TemplateInstantiation &templInst) const;

    // The expr should have size_t type and should evaluate to size
    // of stack memory to be allocated.
    Expr *expr;
};

/** @brief Expression representing a symbol reference in the program */
class SymbolExpr : public Expr {
  public:
    SymbolExpr(Symbol *s, SourcePos p);

    static inline bool classof(SymbolExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == SymbolExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    llvm::Value *GetLValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    const Type *GetLValueType() const;
    Symbol *GetBaseSymbol() const;
    Expr *TypeCheck();
    Expr *Optimize();
    void Print(Indent &indent) const;
    int EstimateCost() const;
    SymbolExpr *Instantiate(TemplateInstantiation &templInst) const;

  private:
    Symbol *symbol;
};

/** @brief Expression representing a function symbol in the program (generally
    used for a function call).
 */
class FunctionSymbolExpr : public Expr {
  public:
    FunctionSymbolExpr(const char *name, const std::vector<Symbol *> &candFuncs, SourcePos pos);
    FunctionSymbolExpr(const char *name, const std::vector<TemplateSymbol *> &candFuncs, const TemplateArgs &templArgs,
                       SourcePos pos);

    static inline bool classof(FunctionSymbolExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == FunctionSymbolExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    Symbol *GetBaseSymbol() const;
    Expr *TypeCheck();
    Expr *Optimize();
    void Print(Indent &indent) const;
    int EstimateCost() const;
    FunctionSymbolExpr *Instantiate(TemplateInstantiation &templInst) const;
    std::pair<llvm::Constant *, bool> GetConstant(const Type *type) const;

    /** Given the types of the function arguments, in the presence of
        function overloading, this method resolves which actual function
        the arguments match best.  If the argCouldBeNULL parameter is
        non-nullptr, each element indicates whether the corresponding argument
        is the number zero, indicating that it could be a nullptr pointer, and
        if argIsConstant is non-nullptr, each element indicates whether the
        corresponding argument is a compile-time constant value.  Both of
        these parameters may be nullptr (for cases where overload resolution
        is being done just given type information without the parameter
        argument expressions being available.  This function returns true
        on success.
     */
    bool ResolveOverloads(SourcePos argPos, const std::vector<const Type *> &argTypes,
                          const std::vector<bool> *argCouldBeNULL = nullptr,
                          const std::vector<bool> *argIsConstant = nullptr);

  private:
    std::vector<Symbol *> getCandidateFunctions(int argCount) const;
    std::vector<Symbol *> getCandidateTemplateFunctions(const std::vector<const Type *> &argTypes) const;
    static int computeOverloadCost(const FunctionType *ftype, const std::vector<const Type *> &argTypes,
                                   const std::vector<bool> *argCouldBeNULL, const std::vector<bool> *argIsConstant,
                                   int *cost);

    /** Name of the function that is being called. */
    std::string name;

    /** All of the functions with the name given in the function call;
        there may be more then one, in which case we need to resolve which
        overload is the best match. */
    std::vector<Symbol *> candidateFunctions;
    std::vector<TemplateSymbol *> candidateTemplateFunctions;
    TemplateArgs templateArgs;

    /** The actual matching function found after overload resolution. */
    Symbol *matchingFunc;

    bool triedToResolve;
    bool unresolvedButDependent;
};

/** @brief A sync statement in the program (waits for all launched tasks before
    proceeding). */
class SyncExpr : public Expr {
  public:
    SyncExpr(SourcePos p) : Expr(p, SyncExprID) {}

    static inline bool classof(SyncExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == SyncExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    Expr *TypeCheck();
    Expr *Optimize();
    void Print(Indent &indent) const;
    int EstimateCost() const;
    SyncExpr *Instantiate(TemplateInstantiation &templInst) const;
};

/** @brief An expression that represents a nullptr pointer. */
class NullPointerExpr : public Expr {
  public:
    NullPointerExpr(SourcePos p) : Expr(p, NullPointerExprID) {}

    static inline bool classof(NullPointerExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == NullPointerExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    Expr *TypeCheck();
    Expr *Optimize();
    std::pair<llvm::Constant *, bool> GetConstant(const Type *type) const;
    void Print(Indent &indent) const;
    int EstimateCost() const;
    NullPointerExpr *Instantiate(TemplateInstantiation &templInst) const;
};

/** An expression representing a "new" expression, used for dynamically
    allocating memory.
*/
class NewExpr : public Expr {
  public:
    NewExpr(int typeQual, const Type *type, Expr *initializer, Expr *count, SourcePos tqPos, SourcePos p);

    static inline bool classof(NewExpr const *) { return true; }
    static inline bool classof(ASTNode const *N) { return N->getValueID() == NewExprID; }

    llvm::Value *GetValue(FunctionEmitContext *ctx) const;
    const Type *GetType() const;
    Expr *TypeCheck();
    Expr *Optimize();
    void Print(Indent &indent) const;
    int EstimateCost() const;
    NewExpr *Instantiate(TemplateInstantiation &templInst) const;

    /** Type of object to allocate storage for. */
    const Type *allocType;
    /** Expression giving the number of elements to allocate, when the
        "new Foo[expr]" form is used.  This may be nullptr, in which case a
        single element of the given type will be allocated. */
    Expr *countExpr;
    /** Optional initializer expression used to initialize the allocated
        memory. */
    Expr *initExpr;
    /** Indicates whether this is a "varying new" or "uniform new"
        (i.e. whether a separate allocation is performed per program
        instance, or whether a single allocation is performed for the
        entire gang of program instances.) */
    bool isVarying;

  private:
    NewExpr(const Type *type, Expr *count, Expr *init, bool isV, SourcePos p);
};

/** This function indicates whether it's legal to convert from fromType to
    toType.  If the optional errorMsgBase and source position parameters
    are provided, then an error message is issued if the type conversion
    isn't possible.
 */
bool CanConvertTypes(const Type *fromType, const Type *toType, const char *errorMsgBase = nullptr,
                     SourcePos pos = SourcePos());

/** This function attempts to convert the given expression to the given
    type, returning a pointer to a new expression that is the result.  If
    the required type conversion is illegal, it returns nullptr and prints an
    error message using the provided string to indicate the context for
    which type conversion was being applied (e.g. "function call
    parameter").
 */
Expr *TypeConvertExpr(Expr *expr, const Type *toType, const char *errorMsgBase);

Expr *MakeBinaryExpr(BinaryExpr::Op o, Expr *a, Expr *b, SourcePos p);

/** Utility routine that emits code to initialize a symbol given an
    initializer expression.

    @param lvalue    Memory location of storage for the symbol's data
    @param symName   Name of symbol (used in error messages)
    @param symType   Type of variable being initialized
    @param initExpr  Expression for the initializer
    @param ctx       FunctionEmitContext to use for generating instructions
    @param pos       Source file position of the variable being initialized
*/
void InitSymbol(AddressInfo *lvalue, const Type *symType, Expr *initExpr, FunctionEmitContext *ctx, SourcePos pos);

bool PossiblyResolveFunctionOverloads(Expr *expr, const Type *type);
} // namespace ispc
