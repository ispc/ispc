/*
  Copyright (c) 2010-2013, Intel Corporation
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

/** @file sym.cpp
    @brief file with definitions for symbol and symbol table classes.
*/

#include "sym.h"
#include "type.h"
#include "util.h"
#include <stdio.h>

///////////////////////////////////////////////////////////////////////////
// Symbol

Symbol::Symbol(const std::string &n, SourcePos p, const Type *t, StorageClass sc) : pos(p), name(n) {
    storagePtr = NULL;
    function = exportedFunction = NULL;
    type = t;
    constValue = NULL;
    storageClass = sc;
    varyingCFDepth = 0;
    parentFunction = NULL;
}

///////////////////////////////////////////////////////////////////////////
// SymbolTable

SymbolTable::SymbolTable() { PushScope(); }

SymbolTable::~SymbolTable() {
    // Otherwise we have mismatched push/pop scopes
    Assert(variables.size() == 1);
    PopScope();
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
}

void SymbolTable::PopScope() {
    Assert(variables.size() > 1);
    freeSymbolMaps.push_back(variables.back());
    variables.pop_back();
}

bool SymbolTable::AddVariable(Symbol *symbol) {
    Assert(symbol != NULL);

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
    return NULL;
}

bool SymbolTable::AddFunction(Symbol *symbol) {
    const FunctionType *ft = CastType<FunctionType>(symbol->type);
    Assert(ft != NULL);
    if (LookupFunction(symbol->name.c_str(), ft) != NULL)
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
        if (matches == NULL)
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
    return NULL;
}

bool SymbolTable::AddType(const char *name, const Type *type, SourcePos pos) {
    const Type *t = LookupType(name);
    if (t != NULL && CastType<UndefinedStructType>(t) == NULL) {
        // If we have a previous declaration of anything other than an
        // UndefinedStructType with this struct name, issue an error.  If
        // we have an UndefinedStructType, then we'll fall through to the
        // code below that adds the definition to the type map.
        Error(pos, "Ignoring redefinition of type \"%s\".", name);
        return false;
    }

    types[name] = type;
    return true;
}

const Type *SymbolTable::LookupType(const char *name) const {
    // Again, search through the type maps backward to get scoping right.
    TypeMapType::const_iterator iter = types.find(name);
    if (iter != types.end())
        return iter->second;
    return NULL;
}

bool SymbolTable::ContainsType(const Type *type) const {
    TypeMapType::const_iterator iter = types.begin();
    while (iter != types.end()) {
        if (iter->second == type) {
            return true;
        }
        iter++;
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

std::vector<std::string> SymbolTable::ClosestTypeMatch(const char *str) const { return closestTypeMatch(str, true); }

std::vector<std::string> SymbolTable::ClosestEnumTypeMatch(const char *str) const {
    return closestTypeMatch(str, false);
}

std::vector<std::string> SymbolTable::closestTypeMatch(const char *str, bool structsVsEnums) const {
    // This follows the same approach as ClosestVariableOrFunctionMatch()
    // above; compute all edit distances, keep the ones shorter than
    // maxDelta, return the first non-empty vector of one or more sets of
    // alternatives with minimal edit distance.
    const int maxDelta = 2;
    std::vector<std::string> matches[maxDelta + 1];

    TypeMapType::const_iterator iter;
    for (iter = types.begin(); iter != types.end(); ++iter) {
        // Skip over either StructTypes or EnumTypes, depending on the
        // value of the structsVsEnums parameter
        bool isEnum = (CastType<EnumType>(iter->second) != NULL);
        if (isEnum && structsVsEnums)
            continue;
        else if (!isEnum && !structsVsEnums)
            continue;

        int dist = StringEditDistance(str, iter->first, maxDelta + 1);
        if (dist <= maxDelta)
            matches[dist].push_back(iter->first);
    }

    for (int i = 0; i <= maxDelta; ++i) {
        if (matches[i].size())
            return matches[i];
    }
    return std::vector<std::string>();
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
    TypeMapType::iterator siter = types.begin();
    while (siter != types.end()) {
        fprintf(stderr, "%*c", depth, ' ');
        fprintf(stderr, "%s -> %s\n", siter->first.c_str(), siter->second->GetString().c_str());
        ++siter;
    }
}

inline int ispcRand() {
#ifdef ISPC_IS_WINDOWS
    return rand();
#else
    return lrand48();
#endif
}

Symbol *SymbolTable::RandomSymbol() {
    int v = ispcRand() % variables.size();
    if (variables[v]->size() == 0)
        return NULL;
    int count = ispcRand() % variables[v]->size();
    SymbolMapType::iterator iter = variables[v]->begin();
    while (count-- > 0) {
        ++iter;
        Assert(iter != variables[v]->end());
    }
    return iter->second;
}

const Type *SymbolTable::RandomType() {
    int count = types.size();
    TypeMapType::iterator iter = types.begin();
    while (count-- > 0) {
        ++iter;
        Assert(iter != types.end());
    }
    return iter->second;
}
