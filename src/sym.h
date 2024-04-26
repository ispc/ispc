/*
  Copyright (c) 2010-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file sym.h

    @brief header file with declarations for symbol and symbol table
    classes.
*/

#pragma once

#include "ctx.h"
#include "decl.h"
#include "ispc.h"

#include <map>

namespace ispc {

class StructType;
class ConstExpr;

/**
   @brief Representation of a program symbol.

   The Symbol class represents a symbol in an ispc program.  Symbols can
   include variables, functions, and named types.  Note that all of the
   members are publically accessible; other code throughout the system
   accesses and modifies the members directly.

   @todo Should we break function symbols into a separate FunctionSymbol
   class and then not have these members that are not applicable for
   function symbols (and vice versa, for non-function symbols)?
 */

class Symbol : public Traceable {
  public:
    /** The Symbol constructor takes the name of the symbol, its
        position in a source file, and its type (if known). */
    Symbol(const std::string &name, SourcePos pos, const Type *t = nullptr, StorageClass sc = SC_NONE);

    SourcePos pos;            /*!< Source file position where the symbol was defined */
    std::string name;         /*!< Symbol's name */
    AddressInfo *storageInfo; /*!< For symbols with storage associated with
                                   them (i.e. variables but not functions),
                                   this member stores an address info: pointer to
                                   its location in memory and its element type.) */
    llvm::Function *function; /*!< For symbols that represent functions,
                                   this stores the LLVM Function value for
                                   the symbol once it has been created. */
    llvm::Function *exportedFunction;
    /*!< For symbols that represent functions with
         'export' qualifiers, this points to the LLVM
         Function for the application-callable version
         of the function. */
    const Type *type;          /*!< The type of the symbol; if not set by the
                                    constructor, this is set after the
                                    declaration around the symbol has been parsed.  */
    ConstExpr *constValue;     /*!< For symbols with const-qualified types, this may store
                                    the symbol's compile-time constant value.  This value may
                                    validly be nullptr for a const-qualified type, however; for
                                    example, the ConstExpr class can't currently represent
                                    struct types.  For cases like these, ConstExpr is nullptr,
                                    though for all const symbols, the value pointed to by the
                                    storageInfo pointer member will be its constant value.  (This
                                    messiness is due to needing an ispc ConstExpr for the early
                                    constant folding optimizations). */
    StorageClass storageClass; /*!< Records the storage class (if any) provided with the
                                    symbol's declaration. */
    int varyingCFDepth;        /*!< This member records the number of levels of nested 'varying'
                                    control flow within which the symbol was declared.  Having
                                    this value available makes it possible to avoid performing
                                    masked stores when modifying the symbol's value when the
                                    store is done at the same 'varying' control flow depth as
                                    the one where the symbol was originally declared. */
    const Function *parentFunction;
    /*!< For symbols that are parameters to functions or are
         variables declared inside functions, this gives the
         function they're in. */
};

/**
   @brief Represents function template.

   TODO: The only reason that it's separate from Symbol is that we trying not to introduce additional
   overhead. Symbol class needs to be refactored to be either a class hierarchy or be implemented as
   a union of different types of symbols.
 */
class TemplateSymbol {
  public:
    TemplateSymbol(const TemplateParms *parms, const std::string &n, const FunctionType *t, StorageClass sc,
                   const SourcePos p, bool isInline, bool inNoInline);

    SourcePos pos;
    const std::string name;
    const FunctionType *type;
    StorageClass storageClass;
    const TemplateParms *templateParms;
    FunctionTemplate *functionTemplate;

    // Inline / noinline attributes.
    // TODO: it's bad idea to store them here, this need to be redesigned.
    // The reason to keep them here for now is that for regular functions it's not stored anywhere in AST,
    // but attached as attrubutes to llvm::Function when it's created. For templates we need to store this
    // information in here and use later when the template is instantiated.
    bool isInline;
    bool isNoInline;
};

/** @brief Symbol table that holds all known symbols during parsing and compilation.

    A single instance of a SymbolTable is stored in the Module class
    (Module::symbolTable); it is created in the Module::Module()
    constructor.  It is then accessed via the global variable Module *\ref m
    throughout the ispc implementation.
 */

class SymbolTable {
  public:
    SymbolTable();
    ~SymbolTable();

    /** The parser calls this method when it enters a new scope in the
        program; this allows us to track variables that shadows others in
        outer scopes with same name as well as to efficiently discard all
        of the variables declared in a particular scope when we exit that
        scope. */
    void PushScope();

    /** For each scope started by a call to SymbolTable::PushScope(), there
        must be a matching call to SymbolTable::PopScope() at the end of
        that scope. */
    void PopScope();

    /** Pop all scopes except the outermost scope. It's needed to clean up SymbolTable in case of any error during
        parsing to avoid assertion in destructor. */
    void PopInnerScopes();

    /** Adds the given variable symbol to the symbol table.
        @param symbol The symbol to be added

        @return true if successful; false if the provided symbol clashes
        with a symbol defined at the same scope.  (Symbols may shaodow
        symbols in outer scopes; a warning is issued in this case, but this
        method still returns true.) */
    bool AddVariable(Symbol *symbol);

    /** Looks for a variable with the given name in the symbol table.  This
        method searches outward from the innermost scope to the outermost,
        returning the first match found.

        @param  name The name of the variable to be searched for.
        @return A pointer to the Symbol, if a match is found.  nullptr if no
        Symbol with the given name is in the symbol table. */
    Symbol *LookupVariable(const char *name);

    /** Adds the given function symbol to the symbol table.
        @param symbol The function symbol to be added.

        @return true if the symbol has been added.  False if another
        function symbol with the same name and function signature is
        already present in the symbol table. */
    bool AddFunction(Symbol *symbol);

    /** Looks for the function or functions with the given name in the
        symbol name.  If a function has been overloaded and multiple
        definitions are present for a given function name, all of them will
        be returned in the provided vector and it's up the the caller to
        resolve which one (if any) to use.  Returns true if any matches
        were found. */
    bool LookupFunction(const char *name, std::vector<Symbol *> *matches = nullptr);

    /** Adds the given function symbol for LLVM intrinsic to the symbol table.
        @param symbol The function symbol to be added.

        @return true if the symbol has been added.  False if another
        function symbol with the same name and function signature is
        already present in the symbol table. */
    bool AddIntrinsics(Symbol *symbol);

    /** Looks for a LLVM intrinsic function in the symbol table.

        @return pointer to matching Symbol; nullptr if none is found. */
    Symbol *LookupIntrinsics(llvm::Function *func);

    /** Looks for a function with the given name and type
        in the symbol table.

        @return pointer to matching Symbol; nullptr if none is found. */
    Symbol *LookupFunction(const char *name, const FunctionType *type);

    /** Adds the given function template to the symbol table.
        @param templ The function template to be added.

        @return true if the template has been added.  False if another
        function template with the same name and function signature is
        already present in the symbol table. */
    bool AddFunctionTemplate(TemplateSymbol *templ);

    /** Looks for the function or functions with the given name in the
        symbol name.  If a function has been overloaded and multiple
        definitions are present for a given function name, all of them will
        be returned in the provided vector and it's up the the caller to
        resolve which one (if any) to use.  Returns true if any matches
        were found. */
    bool LookupFunctionTemplate(const std::string &name, std::vector<TemplateSymbol *> *matches = nullptr);

    /** Looks for a function template with the given name and type
        in the symbol table.

        @return pointer to matching FunctionTemplate; nullptr if none is found. */
    TemplateSymbol *LookupFunctionTemplate(const TemplateParms *templateParmList, const std::string &name,
                                           const FunctionType *type);

    /** Returns all of the functions in the symbol table that match the given
        predicate.

        @param pred A unary predicate that returns true or false, given a Symbol
        pointer, based on whether the symbol should be included in the returned
        set of matches.  It can either be a function, with signature
        <tt>bool pred(const Symbol *s)</tt>, or a unary predicate object with
        an <tt>bool operator()(const Symbol *)</tt> method.

        @param matches Pointer to a vector in which to return the matching
        symbols.
     */
    template <typename Predicate> void GetMatchingFunctions(Predicate pred, std::vector<Symbol *> *matches) const;

    /** Adds the named type to the symbol table.  This is used for both
        struct definitions (where <tt>struct Foo</tt> causes type \c Foo to
        be added to the symbol table) as well as for <tt>typedef</tt>s.
        For structs with forward declarations ("struct Foo;") and are thus
        UndefinedStructTypes, this method replaces these with an actual
        struct definition if one is provided.

        @param name Name of the type to be added
        @param type Type that \c name represents
        @param pos Position in source file where the type was named
        @return true if the named type was successfully added.  False if a type
        with the same name has already been defined.

    */
    bool AddType(const char *name, const Type *type, SourcePos pos);

    /** Looks for a type of the given name in the symbol table.

        @return Pointer to the Type, if found; otherwise nullptr is returned.
    */
    const Type *LookupType(const char *name) const;

    /** Looks for a type of the given name in the most local
        scope in the symbol table. This is useful for determining
        whether a type definition can assume a certain name.

        @return A pointer to the type that was found or null.
     */
    const Type *LookupLocalType(const char *name) const;

    /** Look for a type given a pointer.

        @return True if found, False otherwise.
    */
    bool ContainsType(const Type *type) const;

    /** This method returns zero or more strings with the names of symbols
        in the symbol table that nearly (but not exactly) match the given
        name.  This is useful for issuing informative error methods when
        misspelled identifiers are found a programs.

        @param name String to compare variable and function symbol names against.
        @return vector of zero or more strings that approximately match \c name.
    */
    std::vector<std::string> ClosestVariableOrFunctionMatch(const char *name) const;

    /** This method returns zero or more strings with the names of enum types
        in the symbol table that nearly (but not exactly) match the given
        name. */
    std::vector<std::string> ClosestEnumTypeMatch(const char *name) const;

    /** Prints out the entire contents of the symbol table to standard error.
        (Debugging method). */
    void Print();

    /** Returns a random symbol from the symbol table. (It is not
        guaranteed that it is equally likely to return all symbols). */
    Symbol *RandomSymbol();

    /** Returns a random type from the symbol table. */
    const Type *RandomType();

  private:
    std::vector<std::string> closestTypeMatch(const char *str, bool structsVsEnums) const;

    /** This member variable holds one SymbolMap for each of the current
        active scopes as the program is being parsed.  New maps are added
        and removed from the end of the main vector, so searches for
        symbols start looking at the end of \c variables and work
        backwards.
     */
    typedef std::map<std::string, Symbol *> SymbolMapType;
    std::vector<SymbolMapType *> variables;

    std::vector<SymbolMapType *> freeSymbolMaps;

    /** Function declarations are *not* scoped.  (C99, for example, allows
        an implementation to maintain function declarations in a single
        namespace.)  A STL \c vector is used to store the function symbols
        for a given name since, due to function overloading, a name can
        have multiple function symbols associated with it. */
    typedef std::map<std::string, std::vector<Symbol *>> FunctionMapType;
    FunctionMapType functions;

    /** This maps ISPC symbols for corresponding LLVM intrinsic functions.
     */
    typedef std::map<llvm::Function *, Symbol *> IntrinsicMapType;
    IntrinsicMapType intrinsics;

    /** Function template declarations, as well as function declaration, are
        *not* scoped.  A STL \c vector is used to store the function templates
        for a given name since, due to function overloading, a name can
        have multiple function templates associated with it. */
    typedef std::map<std::string, std::vector<TemplateSymbol *>> FunctionTemplateMapType;
    FunctionTemplateMapType functionTemplates;

    /** Scoped types.
     */
    typedef std::map<std::string, const Type *> TypeMapType;
    std::vector<TypeMapType> types;
};

template <typename Predicate>
void SymbolTable::GetMatchingFunctions(Predicate pred, std::vector<Symbol *> *matches) const {
    // Iterate through all function symbols and apply the given predicate.
    // If it returns true, add the Symbol * to the provided vector.
    FunctionMapType::const_iterator iter;
    for (iter = functions.begin(); iter != functions.end(); ++iter) {
        const std::vector<Symbol *> &syms = iter->second;
        for (unsigned int j = 0; j < syms.size(); ++j) {
            if (pred(syms[j]))
                matches->push_back(syms[j]);
        }
    }
}

} // namespace ispc
