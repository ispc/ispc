/*
  Copyright (c) 2010-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file sym.cpp
    @brief file with definitions for symbol and symbol table classes.
*/

#include "sym.h"
#include "expr.h"
#include "func.h"
#include "type.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <stdio.h>

using namespace ispc;

///////////////////////////////////////////////////////////////////////////
// Symbol

Symbol::Symbol(const std::string &n, SourcePos p, const Type *t, StorageClass sc)
    : pos(p), name(n), storageInfo(nullptr), function(nullptr), exportedFunction(nullptr), type(t), constValue(nullptr),
      storageClass(sc), varyingCFDepth(0), parentFunction(nullptr) {}

///////////////////////////////////////////////////////////////////////////
// TemplateSymbol

TemplateSymbol::TemplateSymbol(const TemplateParms *parms, const std::string &n, const FunctionType *t, StorageClass sc,
                               const SourcePos p, bool inl, bool noinl)
    : pos(p), name(n), type(t), storageClass(sc), templateParms(parms), functionTemplate(nullptr), isInline(inl),
      isNoInline(noinl) {}

///////////////////////////////////////////////////////////////////////////
// SymbolTable

SymbolTable::SymbolTable() { PushScope(); }

SymbolTable::~SymbolTable() {
    // Otherwise we have mismatched push/pop scopes
    Assert(variables.size() == 1);
    PopScope();

    for (auto p : freeSymbolMaps) {
        delete p;
    }

    for (auto const &x : functionTemplates) {
        for (auto *p : x.second) {
            if (p) {
                delete p;
            }
        }
    }
}

void SymbolTable::PushScope() {
    SymbolMapType *sm;
    if (freeSymbolMaps.size() > 0) {
        sm = freeSymbolMaps.back();
        freeSymbolMaps.pop_back();
        sm->erase(sm->begin(), sm->end());
    } else
        sm = new SymbolMapType;

    variables.push_back(sm);
    types.emplace_back();
}

void SymbolTable::PopScope() {
    Assert(variables.size() > 0);
    Assert(types.size() > 0);
    freeSymbolMaps.push_back(variables.back());
    variables.pop_back();
    types.pop_back();
}

void SymbolTable::PopInnerScopes() {
    while (variables.size() > 1) {
        PopScope();
    }
}

bool SymbolTable::AddVariable(Symbol *symbol) {
    Assert(symbol != nullptr);

    // Check to see if a symbol of the same name has already been declared.
    for (int i = (int)variables.size() - 1; i >= 0; --i) {
        SymbolMapType &sm = *(variables[i]);
        if (sm.find(symbol->name) != sm.end()) {
            if (i == (int)variables.size() - 1) {
                // If a symbol of the same name was declared in the
                // same scope, it's an error.
                Error(symbol->pos, "Ignoring redeclaration of symbol \"%s\".", symbol->name.c_str());
                return false;
            } else {
                // Otherwise it's just shadowing something else, which
                // is legal but dangerous..
                Warning(symbol->pos, "Symbol \"%s\" shadows symbol declared in outer scope.", symbol->name.c_str());
                (*variables.back())[symbol->name] = symbol;
                return true;
            }
        }
    }

    // No matches, so go ahead and add it...
    (*variables.back())[symbol->name] = symbol;
    return true;
}

Symbol *SymbolTable::LookupVariable(const char *name) {
    // Note that we iterate through the variables vectors backwards, since
    // we want to search from the innermost scope to the outermost, so that
    // we get the right symbol if we have multiple variables in different
    // scopes that shadow each other.
    for (int i = (int)variables.size() - 1; i >= 0; --i) {
        SymbolMapType &sm = *(variables[i]);
        SymbolMapType::iterator iter = sm.find(name);
        if (iter != sm.end())
            return iter->second;
    }
    return nullptr;
}

bool SymbolTable::AddFunction(Symbol *symbol) {
    const FunctionType *ft = CastType<FunctionType>(symbol->type);
    Assert(ft != nullptr);
    if (LookupFunction(symbol->name.c_str(), ft) != nullptr)
        // A function of the same name and type has already been added to
        // the symbol table
        return false;

    std::vector<Symbol *> &funOverloads = functions[symbol->name];
    funOverloads.push_back(symbol);
    return true;
}

bool SymbolTable::LookupFunction(const char *name, std::vector<Symbol *> *matches) {
    FunctionMapType::iterator iter = functions.find(name);
    if (iter != functions.end()) {
        if (matches == nullptr)
            return true;
        else {
            const std::vector<Symbol *> &funcs = iter->second;
            for (int j = 0; j < (int)funcs.size(); ++j)
                matches->push_back(funcs[j]);
        }
    }
    return matches ? (matches->size() > 0) : false;
}

Symbol *SymbolTable::LookupFunction(const char *name, const FunctionType *type) {
    FunctionMapType::iterator iter = functions.find(name);
    if (iter != functions.end()) {
        std::vector<Symbol *> funcs = iter->second;
        for (int j = 0; j < (int)funcs.size(); ++j) {
            if (Type::Equal(funcs[j]->type, type))
                return funcs[j];
        }
    }
    return nullptr;
}

bool SymbolTable::AddIntrinsics(Symbol *symbol) {
    if (LookupIntrinsics(symbol->function) != nullptr) {
        // A function of the same type has already been added to
        // the symbol table
        return false;
    }
    intrinsics[symbol->function] = symbol;
    return true;
}

Symbol *SymbolTable::LookupIntrinsics(llvm::Function *func) {
    IntrinsicMapType::iterator iter = intrinsics.find(func);
    if (iter != intrinsics.end()) {
        Symbol *funcs = iter->second;
        return funcs;
    }
    return nullptr;
}

bool SymbolTable::AddFunctionTemplate(TemplateSymbol *templ) {
    Assert(templ && templ->templateParms && templ->type);
    if (LookupFunctionTemplate(templ->templateParms, templ->name, templ->type) != nullptr) {
        // A function template of the same name and type has already been added to
        // the symbol table
        return false;
    }

    std::vector<TemplateSymbol *> &funTemplOverloads = functionTemplates[templ->name];
    funTemplOverloads.push_back(templ);
    return true;
}

bool SymbolTable::LookupFunctionTemplate(const std::string &name, std::vector<TemplateSymbol *> *matches) {
    FunctionTemplateMapType::iterator iter = functionTemplates.find(name);
    if (iter != functionTemplates.end()) {
        if (matches == nullptr) {
            return true;
        }
        const std::vector<TemplateSymbol *> &templs = iter->second;
        for (auto templ : templs) {
            matches->push_back(templ);
        }
    }
    return matches ? (matches->size() > 0) : false;
}

TemplateSymbol *SymbolTable::LookupFunctionTemplate(const TemplateParms *templateParmList, const std::string &name,
                                                    const FunctionType *type) {
    // The template declaration matches if:
    // - template paramters list matches
    // - function types match
    FunctionTemplateMapType::iterator iter = functionTemplates.find(name);
    if (iter != functionTemplates.end()) {
        std::vector<TemplateSymbol *> templs = iter->second;
        for (auto templ : templs) {
            if (templateParmList->IsEqual(templ->templateParms) && Type::Equal(templ->type, type)) {
                return templ;
            }
        }
    }
    return nullptr;
}

bool SymbolTable::AddType(const char *name, const Type *type, SourcePos pos) {
    const Type *t = LookupLocalType(name);
    if (t != nullptr && CastType<UndefinedStructType>(t) == nullptr) {
        // If we have a previous declaration of anything other than an
        // UndefinedStructType with this struct name, issue an error.  If
        // we have an UndefinedStructType, then we'll fall through to the
        // code below that adds the definition to the type map.
        Error(pos, "Ignoring redefinition of type \"%s\".", name);
        return false;
    }

    Assert(types.size() > 0);

    types.back()[name] = type;
    return true;
}

const Type *SymbolTable::LookupType(const char *name) const {
    // Again, search through the type maps backward to get scoping right.
    for (std::vector<TypeMapType>::const_reverse_iterator it = types.rbegin(); it != types.rend(); it++) {
        TypeMapType::const_iterator type_it = it->find(name);
        if (type_it != it->end())
            return type_it->second;
    }
    return nullptr;
}

const Type *SymbolTable::LookupLocalType(const char *name) const {

    Assert(types.size() > 0);

    auto result = types.back().find(name);
    if (result == types.back().end())
        return nullptr;
    else
        return result->second;
}

bool SymbolTable::ContainsType(const Type *type) const {
    for (const TypeMapType &typeMap : types) {
        for (const std::pair<const std::string, const Type *> &entry : typeMap) {
            if (entry.second == type)
                return true;
        }
    }
    return false;
}

std::vector<std::string> SymbolTable::ClosestVariableOrFunctionMatch(const char *str) const {
    // This is a little wasteful, but we'll look through all of the
    // variable and function symbols and compute the edit distance from the
    // given string to them.  If the edit distance is under maxDelta, then
    // it goes in the entry of the matches[] array corresponding to its
    // edit distance.
    const int maxDelta = 2;
    std::vector<std::string> matches[maxDelta + 1];

    for (int i = 0; i < (int)variables.size(); ++i) {
        const SymbolMapType &sv = *(variables[i]);
        SymbolMapType::const_iterator iter;
        for (iter = sv.begin(); iter != sv.end(); ++iter) {
            const Symbol *sym = iter->second;
            int dist = StringEditDistance(str, sym->name, maxDelta + 1);
            if (dist <= maxDelta)
                matches[dist].push_back(sym->name);
        }
    }

    FunctionMapType::const_iterator iter;
    for (iter = functions.begin(); iter != functions.end(); ++iter) {
        int dist = StringEditDistance(str, iter->first, maxDelta + 1);
        if (dist <= maxDelta)
            matches[dist].push_back(iter->first);
    }

    // Now, return the first entry of matches[] that is non-empty, if any.
    for (int i = 0; i <= maxDelta; ++i) {
        if (matches[i].size())
            return matches[i];
    }

    // Otherwise, no joy.
    return std::vector<std::string>();
}

std::vector<std::string> SymbolTable::ClosestEnumTypeMatch(const char *str) const {
    // This follows the same approach as ClosestVariableOrFunctionMatch()
    // above; compute all edit distances, keep the ones shorter than
    // maxDelta, return the first non-empty vector of one or more sets of
    // alternatives with minimal edit distance.
    const int maxDelta = 2;
    std::array<std::vector<std::string>, maxDelta + 1> matches;

    for (const TypeMapType &typeMap : types) {
        for (const std::pair<const std::string, const Type *> &entry : typeMap) {
            // Skip over either StructTypes or EnumTypes, depending on the
            // value of the structsVsEnums parameter
            bool isEnum = (CastType<EnumType>(entry.second) != nullptr);
            if (!isEnum)
                continue;

            int dist = StringEditDistance(str, entry.first, maxDelta + 1);
            if (dist <= maxDelta)
                matches[dist].push_back(entry.first);
        }
    }

    auto predicate = [](const std::vector<std::string> &set) { return set.empty(); };

    auto result = std::find_if_not(matches.begin(), matches.end(), predicate);
    if (result == matches.end())
        return std::vector<std::string>();
    else
        return *result;
}

void SymbolTable::Print() {
    int depth = 0;
    fprintf(stderr, "Variables:\n----------------\n");
    for (int i = 0; i < (int)variables.size(); ++i) {
        SymbolMapType &sm = *(variables[i]);
        SymbolMapType::iterator iter;
        for (iter = sm.begin(); iter != sm.end(); ++iter) {
            fprintf(stderr, "%*c", depth, ' ');
            Symbol *sym = iter->second;
            fprintf(stderr, "%s [%s]", sym->name.c_str(), sym->type->GetString().c_str());
        }
        fprintf(stderr, "\n");
        depth += 4;
    }

    fprintf(stderr, "Functions:\n----------------\n");
    FunctionMapType::iterator fiter = functions.begin();
    while (fiter != functions.end()) {
        fprintf(stderr, "%s\n", fiter->first.c_str());
        std::vector<Symbol *> &syms = fiter->second;
        for (unsigned int j = 0; j < syms.size(); ++j)
            fprintf(stderr, "    %s\n", syms[j]->type->GetString().c_str());
        ++fiter;
    }

    depth = 0;
    fprintf(stderr, "Named types:\n---------------\n");
    for (const TypeMapType &typeMap : types) {
        for (const std::pair<const std::string, const Type *> &entry : typeMap) {
            fprintf(stderr, "%*c", depth, ' ');
            fprintf(stderr, "%s -> %s\n", entry.first.c_str(), entry.second->GetString().c_str());
        }
        fprintf(stderr, "\n");
        depth += 4;
    }
}

inline int ispcRand() {
#ifdef ISPC_HOST_IS_WINDOWS
    return rand();
#else
    return lrand48();
#endif
}

Symbol *SymbolTable::RandomSymbol() {
    int v = ispcRand() % variables.size();
    if (variables[v]->size() == 0)
        return nullptr;
    int count = ispcRand() % variables[v]->size();
    SymbolMapType::iterator iter = variables[v]->begin();
    while (count-- > 0) {
        ++iter;
        Assert(iter != variables[v]->end());
    }
    return iter->second;
}

const Type *SymbolTable::RandomType() {

    int randomScopeIndex = ispcRand() % types.size();

    int randomTypeIndex = ispcRand() % types[randomScopeIndex].size();

    return std::next(types[randomScopeIndex].cbegin(), randomTypeIndex)->second;
}
