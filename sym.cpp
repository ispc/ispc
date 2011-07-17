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

/** @file sym.cpp
    @brief file with definitions for symbol and symbol table classes. 
*/

#include "sym.h"
#include "type.h"
#include "util.h"
#include <stdio.h>

///////////////////////////////////////////////////////////////////////////
// Symbol

Symbol::Symbol(const std::string &n, SourcePos p, const Type *t) 
  : pos(p), name(n) {
    storagePtr = NULL;
    function = NULL;
    type = t;
    constValue = NULL;
    isStatic = false;
    varyingCFDepth = 0;
}


std::string
Symbol::MangledName() const {
    return name + type->Mangle();
}


///////////////////////////////////////////////////////////////////////////
// SymbolTable

SymbolTable::SymbolTable() {
    PushScope();
}


SymbolTable::~SymbolTable() {
    // Otherwise we have mismatched push/pop scopes
    assert(variables.size() == 1 && types.size() == 1);
    PopScope();
}

void
SymbolTable::PushScope() { 
    variables.push_back(new std::vector<Symbol *>); 
    types.push_back(new TypeMapType);
}


void
SymbolTable::PopScope() { 
    // FIXME: delete Symbols in variables vector<>...
    assert(variables.size() > 1);
    delete variables.back();
    variables.pop_back();
    assert(types.size() > 1);
    delete types.back();
    types.pop_back();
}


bool
SymbolTable::AddVariable(Symbol *symbol) {
    assert(symbol != NULL);

    // Check to see if a symbol of the same name has already been declared.
    for (int i = (int)variables.size() - 1; i >= 0; --i) {
        std::vector<Symbol *> &sv = *(variables[i]);
        for (int j = (int)sv.size() - 1; j >= 0; --j) {
            if (sv[j]->name == symbol->name) {
                if (i == (int)variables.size()-1) {
                    // If a symbol of the same name was declared in the
                    // same scope, it's an error.
                    Error(symbol->pos, "Ignoring redeclaration of symbol \"%s\".", 
                          symbol->name.c_str());
                    return false;
                }
                else {
                    // Otherwise it's just shadowing something else, which
                    // is legal but dangerous..
                    Warning(symbol->pos, 
                            "Symbol \"%s\" shadows symbol declared in outer scope.",
                            symbol->name.c_str());
                    variables.back()->push_back(symbol);
                    return true;
                }
            }
        }
    }

    // No matches, so go ahead and add it...
    variables.back()->push_back(symbol);
    return true;
}


Symbol *
SymbolTable::LookupVariable(const char *name) {
    // Note that we iterate through the variables vectors backwards, sinec
    // we want to search from the innermost scope to the outermost, so that
    // we get the right symbol if we have multiple variables in different
    // scopes that shadow each other.
    std::vector<std::vector<Symbol *> *>::reverse_iterator liter = variables.rbegin();
    while (liter != variables.rend()) {
        std::vector<Symbol *> &sv = *(*liter);
        for (int i = (int)sv.size() - 1; i >= 0; --i)
            if (sv[i]->name == name) 
                return sv[i];
        ++liter;
    }
    return NULL;
}


bool
SymbolTable::AddFunction(Symbol *symbol) {
    const FunctionType *ft = dynamic_cast<const FunctionType *>(symbol->type);
    assert(ft != NULL);
    if (LookupFunction(symbol->name.c_str(), ft) != NULL)
        // A function of the same name and type has already been added to
        // the symbol table
        return false;

    functions[symbol->name].push_back(symbol);
    return true;
}


std::vector<Symbol *> *
SymbolTable::LookupFunction(const char *name) {
    if (functions.find(name) != functions.end())
        return &functions[name];
    return NULL;
}


Symbol *
SymbolTable::LookupFunction(const char *name, const FunctionType *type) {
    if (functions.find(name) == functions.end())
        return NULL;

    std::vector<Symbol *> &funcs = functions[name];
    for (unsigned int i = 0; i < funcs.size(); ++i)
        if (Type::Equal(funcs[i]->type, type))
            return funcs[i];
    return NULL;
}


bool
SymbolTable::AddType(const char *name, const Type *type, SourcePos pos) {
    // Like AddVariable(), we go backwards through the type maps, working
    // from innermost scope to outermost.
    for (int i = types.size()-1; i >= 0; --i) {
        TypeMapType &sm = *(types[i]);
        if (sm.find(name) != sm.end()) {
            if (i == (int)types.size() - 1) {
                Error(pos, "Ignoring redefinition of type \"%s\".", name);
                return false;
            }
            else {
                Warning(pos, "Type \"%s\" shadows type declared in outer scope.", name);
                TypeMapType &sm = *(types.back());
                sm[name] = type;
                return true;
            }
        }
    }

    TypeMapType &sm = *(types.back());
    sm[name] = type;
    return true;
}


const Type *
SymbolTable::LookupType(const char *name) const {
    // Again, search through the type maps backward to get scoping right.
    for (int i = types.size()-1; i >= 0; --i) {
        TypeMapType &sm = *(types[i]);
        if (sm.find(name) != sm.end())
            return sm[name];
    }
    return NULL;
}


std::vector<std::string>
SymbolTable::ClosestVariableOrFunctionMatch(const char *str) const {
    // This is a little wasteful, but we'll look through all of the
    // variable and function symbols and compute the edit distance from the
    // given string to them.  If the edit distance is under maxDelta, then
    // it goes in the entry of the matches[] array corresponding to its
    // edit distance.
    const int maxDelta = 2;
    std::vector<std::string> matches[maxDelta+1];

    for (int i = 0; i < (int)variables.size(); ++i) {
        std::vector<Symbol *> &sv = *(variables[i]);
        for (int j = 0; j < (int)sv.size(); ++j) {
            int dist = StringEditDistance(str, sv[j]->name, maxDelta+1);
            if (dist <= maxDelta)
                matches[dist].push_back(sv[j]->name);
        }
    }

    std::map<std::string, std::vector<Symbol *> >::const_iterator iter;
    for (iter = functions.begin(); iter != functions.end(); ++iter) {
        int dist = StringEditDistance(str, iter->first, maxDelta+1);
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


std::vector<std::string>
SymbolTable::ClosestTypeMatch(const char *str) const {
    return closestTypeMatch(str, true);
}


std::vector<std::string>
SymbolTable::ClosestEnumTypeMatch(const char *str) const {
    return closestTypeMatch(str, false);
}


std::vector<std::string>
SymbolTable::closestTypeMatch(const char *str, bool structsVsEnums) const {
    // This follows the same approach as ClosestVariableOrFunctionMatch()
    // above; compute all edit distances, keep the ones shorter than
    // maxDelta, return the first non-empty vector of one or more sets of
    // alternatives with minimal edit distance.
    const int maxDelta = 2;
    std::vector<std::string> matches[maxDelta+1];

    for (unsigned int i = 0; i < types.size(); ++i) {
        TypeMapType::const_iterator iter;
        for (iter = types[i]->begin(); iter != types[i]->end(); ++iter) {
            // Skip over either StructTypes or EnumTypes, depending on the
            // value of the structsVsEnums parameter
            bool isEnum = (dynamic_cast<const EnumType *>(iter->second) != NULL);
            if (isEnum && structsVsEnums)
                continue;
            else if (!isEnum && !structsVsEnums)
                continue;

            int dist = StringEditDistance(str, iter->first, maxDelta+1);
            if (dist <= maxDelta)
                matches[dist].push_back(iter->first);
        }
    }

    for (int i = 0; i <= maxDelta; ++i) {
        if (matches[i].size())
            return matches[i];
    }
    return std::vector<std::string>();
}


void
SymbolTable::Print() {
    int depth = 0;
    fprintf(stderr, "Variables:\n----------------\n");
    std::vector<std::vector<Symbol *> *>::iterator liter = variables.begin();
    while (liter != variables.end()) {
        fprintf(stderr, "%*c", depth, ' ');
        std::vector<Symbol *>::iterator siter = (*liter)->begin();
        while (siter != (*liter)->end()) {
            fprintf(stderr, "%s [%s]", (*siter)->name.c_str(), 
                    (*siter)->type->GetString().c_str());
            ++siter;
        }
        ++liter;
        fprintf(stderr, "\n");
        depth += 4;
    }

    fprintf(stderr, "Functions:\n----------------\n");
    std::map<std::string, std::vector<Symbol *> >::iterator fiter;
    fiter = functions.begin();
    while (fiter != functions.end()) {
        fprintf(stderr, "%s\n", fiter->first.c_str());
        std::vector<Symbol *> &syms = fiter->second;
        for (unsigned int i = 0; i < syms.size(); ++i)
            fprintf(stderr, "    %s\n", syms[i]->type->GetString().c_str());
        ++fiter;
    }

    depth = 0;
    fprintf(stderr, "Named types:\n---------------\n");
    for (unsigned int i = 0; i < types.size(); ++i) {
        TypeMapType &sm = *types[i];
        TypeMapType::iterator siter = sm.begin();
        while (siter != sm.end()) {
            fprintf(stderr, "%*c", depth, ' ');
            fprintf(stderr, "%s -> %s\n", siter->first.c_str(),
                    siter->second->GetString().c_str());
            ++siter;
        }
        depth += 4;
    }
}
