/*
  Copyright (c) 2011-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file func.h
    @brief Representation of a function in a source file.
*/

#pragma once

#include "ast.h"
#include "ispc.h"
#include "type.h"

#include <unordered_map>
#include <vector>

namespace ispc {

class Function {
  public:
    Function(Symbol *sym, Stmt *code);
    Function(Symbol *sym, Stmt *code, Symbol *maskSymbol, std::vector<Symbol *> &args);

    const Type *GetReturnType() const;
    const FunctionType *GetType() const;

    /** Generate LLVM IR for the function into the current module. */
    void GenerateIR();

    void Print() const;
    void Print(Indent &indent) const;
    bool IsStdlibSymbol() const;

  private:
    enum class DebugPrintPoint { Initial, AfterTypeChecking, AfterOptimization };
    void debugPrintHelper(DebugPrintPoint dumpPoint);
    void typeCheckAndOptimize();

    void emitCode(FunctionEmitContext *ctx, llvm::Function *function, SourcePos firstStmtPos);

    Symbol *sym;
    std::vector<Symbol *> args;
    Stmt *code;
    Symbol *maskSymbol;
    Symbol *threadIndexSym, *threadCountSym;
    Symbol *taskIndexSym, *taskCountSym;
    Symbol *taskIndexSym0, *taskCountSym0;
    Symbol *taskIndexSym1, *taskCountSym1;
    Symbol *taskIndexSym2, *taskCountSym2;
};

// Represents a single template parameter, which can either be a type (TemplateTypeParmType) or a non-type
// (Symbol).
class TemplateParam : public Traceable {
  public:
    enum class ParamType { Type, NonType };

  private:
    ParamType paramType;
    union {
        const TemplateTypeParmType *typeParam;
        Symbol *nonTypeParam;
    };

    std::string name;
    SourcePos pos;

  public:
    TemplateParam(const TemplateTypeParmType *tp);
    TemplateParam(Symbol *ntp);

    // Checks if the parameter is a type parameter.
    bool IsTypeParam() const;
    // Checks if the parameter is a non-type parameter.
    bool IsNonTypeParam() const;
    // Compares two `TemplateParam` instances for equality.
    bool IsEqual(const TemplateParam &other) const;
    // Gets the name of the parameter.
    std::string GetName() const;
    // Returns type parameter.
    const TemplateTypeParmType *GetTypeParam() const;
    // Returns type parameter.
    Symbol *GetNonTypeParam() const;
    // Returns the source position associated with this template parameter.
    SourcePos GetSourcePos() const;
};

// A helper class to manage template parameters list and clean up allocated memory.
class TemplateParms : public Traceable {
  public:
    TemplateParms();
    void Add(const TemplateParam *);
    size_t GetCount() const;
    const TemplateParam *operator[](size_t i) const;
    const TemplateParam *operator[](size_t i);
    bool IsEqual(const TemplateParms *p) const;

  private:
    std::vector<const TemplateParam *> parms;
};

// Represents a single argument in a template instantiation. This can either be a type
// or a non-type (constant or symbol expression).
class TemplateArg : public Traceable {
  public:
    enum class ArgType { Type, NonType };

  private:
    ArgType argType;
    union {
        const Type *type;
        const Expr *expr;
    };
    SourcePos pos;

  public:
    TemplateArg(const Type *t, SourcePos pos);
    TemplateArg(const Expr *c, SourcePos pos);

    // Returns ISPC type of the argument.
    const Type *GetAsType() const;
    // Returns const expression if this argument is a ConstExpr or SymbolExpr with constValue.
    // Note that SymbolExpr may not have constValue until it has been instantiated during template function call.
    // See: FunctionSymbolExpr::Instantiate()
    const ConstExpr *GetAsConstExpr() const;
    // Returns expression if this argument is a non-type, nullptr otherwise.
    const Expr *GetAsExpr() const;
    // Returns the source position associated with this template argument.
    SourcePos GetPos() const;
    // Produces a string representation of the argument.
    std::string GetString() const;
    // Returns `true` if this argument is a non-type.
    bool IsNonType() const;
    // Returns `true` if this argument is a Type.
    bool IsType() const;
    // Mangles the stored type to a string representation.
    std::string Mangle() const;
    // Transforms the stored type to its varying equivalent.
    void SetAsVaryingType();
    // Operator support
    bool operator==(const TemplateArg &other) const;
};

enum class TemplateInstantiationKind { Implicit, Explicit, Specialization };
class FunctionTemplate {
  public:
    FunctionTemplate(TemplateSymbol *sym, Stmt *code);
    ~FunctionTemplate();
    std::string GetName() const;
    const TemplateParms *GetTemplateParms() const;
    const FunctionType *GetFunctionType() const;
    StorageClass GetStorageClass();

    Symbol *LookupInstantiation(const TemplateArgs &tArgs);
    Symbol *AddInstantiation(const TemplateArgs &tArgs, TemplateInstantiationKind kind, bool isInline, bool isNoInline);
    Symbol *AddSpecialization(const FunctionType *ftype, const TemplateArgs &tArgs, bool isInline, bool isNoInline,
                              SourcePos pos);

    // Generate code for instantiations and specializations.
    void GenerateIR() const;

    void Print() const;
    void Print(Indent &indent) const;
    bool IsStdlibSymbol() const;

  private:
    TemplateSymbol *sym;
    std::vector<Symbol *> args;
    Stmt *code;
    Symbol *maskSymbol;

    std::vector<std::pair<TemplateArgs *, Symbol *>> instantiations;
};

// A helper class to drive function instantiation, it provides the following:
// - mapping of the symbols in the template to the symbols in the instantiation
// - type instantiation
class TemplateInstantiation {
  public:
    TemplateInstantiation(const TemplateParms &typeParms, const TemplateArgs &tArgs, TemplateInstantiationKind kind,
                          bool IsInline, bool IsNoInline);
    const Type *InstantiateType(const std::string &name);
    Symbol *InstantiateSymbol(Symbol *sym);
    Symbol *InstantiateTemplateSymbol(TemplateSymbol *sym);
    void SetFunction(Function *func);

    void AddArgument(std::string paramName, TemplateArg arg);

  private:
    // Function Symbol of the instantiation.
    Symbol *functionSym;
    // Mapping of the symbols in the template to correspoding symbols in the instantiation.
    std::unordered_map<Symbol *, Symbol *> symMap;
    // Mapping of template parameter names to the template args in the instantiation.
    std::unordered_map<std::string, const TemplateArg *> argsMap;
    // Template arguments in the order of the template parameters.
    TemplateArgs templateArgs;
    // Kind of instantiation (explicit, implicit, specialization).
    TemplateInstantiationKind kind;

    bool isInline;
    bool isNoInline;

    llvm::Function *createLLVMFunction(Symbol *functionSym);
};

} // namespace ispc
