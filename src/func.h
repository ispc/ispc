/*
  Copyright (c) 2011-2023, Intel Corporation

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

// A helper class to manage template parameters list.
class TemplateParms : public Traceable {
  public:
    TemplateParms();
    void Add(const TemplateTypeParmType *);
    size_t GetCount() const;
    const TemplateTypeParmType *operator[](size_t i) const;
    bool IsEqual(const TemplateParms *p) const;

  private:
    std::vector<const TemplateTypeParmType *> parms;
};

class TemplateArgs {
  public:
    TemplateArgs(const std::vector<std::pair<const Type *, SourcePos>> &args);
    bool IsEqual(TemplateArgs &otherArgs) const;

    std::vector<std::pair<const Type *, SourcePos>> args;
};

class FunctionTemplate {
  public:
    FunctionTemplate(TemplateSymbol *sym, Stmt *code);
    ~FunctionTemplate();
    std::string GetName() const;
    const TemplateParms *GetTemplateParms() const;
    const FunctionType *GetFunctionType() const;

    Symbol *LookupInstantiation(const std::vector<std::pair<const Type *, SourcePos>> &types);
    Symbol *AddInstantiation(const std::vector<std::pair<const Type *, SourcePos>> &types);
    Symbol *AddSpecialization(const FunctionType *ftype, const std::vector<std::pair<const Type *, SourcePos>> &types,
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
    TemplateInstantiation(const TemplateParms &typeParms,
                          const std::vector<std::pair<const Type *, SourcePos>> &typeArgs);
    const Type *InstantiateType(const std::string &name);
    Symbol *InstantiateSymbol(Symbol *sym);
    Symbol *InstantiateTemplateSymbol(TemplateSymbol *sym);
    void SetFunction(Function *func);

    void AddArgument(std::string paramName, const Type *argType);

  private:
    // Function Symbol of the instantiation.
    Symbol *functionSym;
    // Mapping of the symbols in the template to correspoding symbols in the instantiation.
    std::unordered_map<Symbol *, Symbol *> symMap;
    // Mapping of template parameter names to the types in the instantiation.
    std::unordered_map<std::string, const Type *> argsMap;
    // Template arguments in the order of the template parameters.
    std::vector<const Type *> templateArgs;

    llvm::Function *createLLVMFunction(Symbol *functionSym, bool isInline, bool isNoInline);
};

} // namespace ispc
