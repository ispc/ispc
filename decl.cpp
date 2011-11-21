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

/** @file decl.cpp
    @brief Implementations of classes related to turning declarations into 
           symbols and types.
*/

#include "decl.h"
#include "util.h"
#include "module.h"
#include "sym.h"
#include "type.h"
#include "stmt.h"
#include "expr.h"
#include <stdio.h>
#include <llvm/Module.h>

static const Type *
lApplyTypeQualifiers(int typeQualifiers, const Type *type, SourcePos pos) {
    if (type == NULL)
        return NULL;

    // Account for 'unsigned' and 'const' qualifiers in the type
    if ((typeQualifiers & TYPEQUAL_UNSIGNED) != 0) {
        const Type *unsignedType = type->GetAsUnsignedType();
        if (unsignedType != NULL)
            type = unsignedType;
        else
            Error(pos, "\"unsigned\" qualifier is illegal with \"%s\" type.",
              type->GetString().c_str());
    }
    if ((typeQualifiers & TYPEQUAL_CONST) != 0)
        type = type->GetAsConstType();

    // if uniform/varying is specified explicitly, then go with that
    if (dynamic_cast<const FunctionType *>(type) == NULL) {
        if ((typeQualifiers & TYPEQUAL_UNIFORM) != 0)
            type = type->GetAsUniformType();
        else if ((typeQualifiers & TYPEQUAL_VARYING) != 0)
            type = type->GetAsVaryingType();
        else {
            // otherwise, structs are uniform by default and everything
            // else is varying by default
            if (dynamic_cast<const StructType *>(type->GetBaseType()) != NULL)
                type = type->GetAsUniformType();
            else
                type = type->GetAsVaryingType();
        }
    }

    return type;
}


///////////////////////////////////////////////////////////////////////////
// DeclSpecs

DeclSpecs::DeclSpecs(const Type *t, StorageClass sc, int tq) {
    baseType = t;
    storageClass = sc;
    typeQualifiers = tq;
    soaWidth = 0;
    vectorSize = 0;
}


const Type *
DeclSpecs::GetBaseType(SourcePos pos) const {
    const Type *bt = baseType;
    if (vectorSize > 0) {
        const AtomicType *atomicType = dynamic_cast<const AtomicType *>(bt);
        if (atomicType == NULL) {
            Error(pos, "Only atomic types (int, float, ...) are legal for vector "
                  "types.");
            return NULL;
        }
        bt = new VectorType(atomicType, vectorSize);
    }

    return lApplyTypeQualifiers(typeQualifiers, bt, pos);
}


void
DeclSpecs::Print() const {
    if (storageClass == SC_EXTERN)   printf("extern ");
    if (storageClass == SC_EXTERN_C) printf("extern \"C\" ");
    if (storageClass == SC_EXPORT)   printf("export ");
    if (storageClass == SC_STATIC)   printf("static ");
    if (storageClass == SC_TYPEDEF)  printf("typedef ");

    if (soaWidth > 0) printf("soa<%d> ", soaWidth);

    if (typeQualifiers & TYPEQUAL_INLINE)    printf("inline ");
    if (typeQualifiers & TYPEQUAL_CONST)     printf("const ");
    if (typeQualifiers & TYPEQUAL_UNIFORM)   printf("uniform ");
    if (typeQualifiers & TYPEQUAL_VARYING)   printf("varying ");
    if (typeQualifiers & TYPEQUAL_TASK)      printf("task ");
    if (typeQualifiers & TYPEQUAL_REFERENCE) printf("reference ");
    if (typeQualifiers & TYPEQUAL_UNSIGNED)  printf("unsigned ");

    printf("%s", baseType->GetString().c_str());

    if (vectorSize > 0) printf("<%d>", vectorSize);
}


///////////////////////////////////////////////////////////////////////////
// Declarator

Declarator::Declarator(DeclaratorKind dk, SourcePos p) 
    : pos(p), kind(dk) { 
    child = NULL;
    typeQualifiers = 0;
    arraySize = -1;
    sym = NULL;
    initExpr = NULL;
}


void
Declarator::InitFromDeclSpecs(DeclSpecs *ds) {
    const Type *t = GetType(ds);
    Symbol *sym = GetSymbol();
    if (sym != NULL) {
        sym->type = t;
        sym->storageClass = ds->storageClass;
    }
}


Symbol *
Declarator::GetSymbol() {
    Declarator *d = this;
    while (d->child != NULL)
        d = d->child;
    return d->sym;
}


void
Declarator::Print() const {
    printf("%s", sym->name.c_str());
    if (initExpr != NULL) {
        printf(" = (");
        initExpr->Print();
        printf(")");
    }
    pos.Print();
}


void
Declarator::GetFunctionInfo(DeclSpecs *ds, Symbol **funSym, 
                            std::vector<Symbol *> *funArgs) {
    // Get the symbol for the function from the symbol table.  (It should
    // already have been added to the symbol table by AddGlobal() by the
    // time we get here.)
    const FunctionType *type = 
        dynamic_cast<const FunctionType *>(GetType(ds));
    if (type == NULL)
        return;
    Symbol *declSym = GetSymbol();
    assert(declSym != NULL);
    *funSym = m->symbolTable->LookupFunction(declSym->name.c_str(), type);
    if (*funSym != NULL)
        // May be NULL due to error earlier in compilation
        (*funSym)->pos = pos;

    for (unsigned int i = 0; i < functionArgs.size(); ++i) {
        Declaration *pdecl = functionArgs[i];
        assert(pdecl->declarators.size() == 1);
        funArgs->push_back(pdecl->declarators[0]->GetSymbol());
    }
}


const Type *
Declarator::GetType(const Type *base, DeclSpecs *ds) const {
    bool hasUniformQual = ((typeQualifiers & TYPEQUAL_UNIFORM) != 0);
    bool hasVaryingQual = ((typeQualifiers & TYPEQUAL_VARYING) != 0);
    bool isTask =         ((typeQualifiers & TYPEQUAL_TASK) != 0);
    bool isReference =    ((typeQualifiers & TYPEQUAL_REFERENCE) != 0);
    bool isConst =        ((typeQualifiers & TYPEQUAL_CONST) != 0);

    if (hasUniformQual && hasVaryingQual) {
        Error(pos, "Can't provide both \"uniform\" and \"varying\" qualifiers.");
        return NULL;
    }
    if (kind != DK_FUNCTION && isTask)
        Error(pos, "\"task\" qualifier illegal in variable declaration.");

    const Type *type = base;
    switch (kind) {
    case DK_BASE:
        assert(typeQualifiers == 0);
        assert(child == NULL);
        return type;

    case DK_POINTER:
        type = new PointerType(type, hasUniformQual, isConst);
        if (child)
            return child->GetType(type, ds);
        else
            return type;
        break;

    case DK_ARRAY:
        type = new ArrayType(type, arraySize);
        if (child)
            return child->GetType(type, ds);
        else
            return type;
        break;

    case DK_FUNCTION: {
        std::vector<const Type *> args;
        std::vector<std::string> argNames;
        std::vector<ConstExpr *> argDefaults;
        std::vector<SourcePos> argPos;

        // Loop over the function arguments and get names and types for
        // each one in the args and argNames arrays
        for (unsigned int i = 0; i < functionArgs.size(); ++i) {
            Declaration *d = functionArgs[i];
            char buf[32];
            Symbol *sym;
            if (d->declarators.size() == 0) {
                // function declaration like foo(float), w/o a name for
                // the parameter
                sprintf(buf, "__anon_parameter_%d", i);
                sym = new Symbol(buf, pos);
                sym->type = d->declSpecs->GetBaseType(pos);
            }
            else {
                sym = d->declarators[0]->GetSymbol();
                if (sym == NULL) {
                    sprintf(buf, "__anon_parameter_%d", i);
                    sym = new Symbol(buf, pos);
                    sym->type = d->declarators[0]->GetType(d->declSpecs);
                }
            }

            const ArrayType *at = dynamic_cast<const ArrayType *>(sym->type);
            if (at != NULL) {
                // Arrays are passed by reference, so convert array
                // parameters to be references here.
                sym->type = new ReferenceType(sym->type, sym->type->IsConstType());

                // Make sure there are no unsized arrays (other than the
                // first dimension) in function parameter lists.
                at = dynamic_cast<const ArrayType *>(at->GetElementType());
                while (at != NULL) {
                    if (at->GetElementCount() == 0)
                        Error(sym->pos, "Arrays with unsized dimensions in "
                              "dimensions after the first one are illegal in "
                              "function parameter lists.");
                    at = dynamic_cast<const ArrayType *>(at->GetElementType());
                }
            }

            args.push_back(sym->type);
            argNames.push_back(sym->name);
            argPos.push_back(sym->pos);

            ConstExpr *init = NULL;
            if (d->declarators.size()) {
                Declarator *decl = d->declarators[0];
                while (decl->child != NULL) {
                    assert(decl->initExpr == NULL);
                    decl = decl->child;
                }

                if (decl->initExpr != NULL &&
                    (decl->initExpr = decl->initExpr->TypeCheck()) != NULL &&
                    (decl->initExpr = decl->initExpr->Optimize()) != NULL &&
                    (init = dynamic_cast<ConstExpr *>(decl->initExpr)) == NULL) {
                    Error(decl->initExpr->pos, "Default value for parameter "
                          "\"%s\" must be a compile-time constant.", 
                          sym->name.c_str());
                }
            }
            argDefaults.push_back(init);
        }

        if (isReference) {
            Error(pos, "Function return types can't be reference types.");
            return NULL;
        }

        const Type *returnType = type;
        if (returnType == NULL) {
            Error(pos, "No return type provided in function declaration.");
            return NULL;
        }

        bool isExported = ds && (ds->storageClass == SC_EXPORT);
        bool isExternC =  ds && (ds->storageClass == SC_EXTERN_C);
        bool isTask =     ds && ((ds->typeQualifiers & TYPEQUAL_TASK) != 0);
        Type *functionType = 
            new FunctionType(returnType, args, pos, argNames, argDefaults,
                             argPos, isTask, isExported, isExternC);
        return child->GetType(functionType, ds);
    }
    default:
        FATAL("Unexpected decl kind");
        return NULL;
    }

#if 0
            // Make sure we actually have an array of structs ..
            const StructType *childStructType = 
                dynamic_cast<const StructType *>(childType);
            if (childStructType == NULL) {
                Error(pos, "Illegal to provide soa<%d> qualifier with non-struct "
                      "type \"%s\".", soaWidth, childType->GetString().c_str());
                return new ArrayType(childType, arraySize == -1 ? 0 : arraySize);
            }
            else if ((soaWidth & (soaWidth - 1)) != 0) {
                Error(pos, "soa<%d> width illegal.  Value must be power of two.",
                      soaWidth);
                return NULL;
            }
            else if (arraySize != -1 && (arraySize % soaWidth) != 0) {
                Error(pos, "soa<%d> width must evenly divide array size %d.",
                      soaWidth, arraySize);
                return NULL;
            }
            return new SOAArrayType(childStructType, arraySize == -1 ? 0 : arraySize,
                                    soaWidth);
#endif
}


const Type *
Declarator::GetType(DeclSpecs *ds) const {
    const Type *baseType = ds->GetBaseType(pos);
    const Type *type = GetType(baseType, ds);

    if ((ds->typeQualifiers & TYPEQUAL_REFERENCE) != 0) {
        bool hasConstQual = ((ds->typeQualifiers & TYPEQUAL_CONST) != 0);
        type = new ReferenceType(type, hasConstQual);
    }

    return type;
}


///////////////////////////////////////////////////////////////////////////
// Declaration

Declaration::Declaration(DeclSpecs *ds, std::vector<Declarator *> *dlist) {
    declSpecs = ds;
    if (dlist != NULL)
        declarators = *dlist;
    for (unsigned int i = 0; i < declarators.size(); ++i)
        if (declarators[i] != NULL)
            declarators[i]->InitFromDeclSpecs(declSpecs);
}


Declaration::Declaration(DeclSpecs *ds, Declarator *d) {
    declSpecs = ds;
    if (d) {
        d->InitFromDeclSpecs(ds);
        declarators.push_back(d);
    }
}


std::vector<VariableDeclaration>
Declaration::GetVariableDeclarations() const {
    assert(declSpecs->storageClass != SC_TYPEDEF);
    std::vector<VariableDeclaration> vars;

    for (unsigned int i = 0; i < declarators.size(); ++i) {
        if (declarators[i] == NULL)
            continue;
        Declarator *decl = declarators[i];
        if (decl == NULL || decl->kind == DK_FUNCTION) 
            continue;

        Symbol *sym = decl->GetSymbol();
        m->symbolTable->AddVariable(sym);

        vars.push_back(VariableDeclaration(sym, decl->initExpr));
    }
    return vars;
}


void
Declaration::Print() const {
    printf("Declaration: specs [");
    declSpecs->Print();
    printf("], declarators [");
    for (unsigned int i = 0 ; i < declarators.size(); ++i) {
        declarators[i]->Print();
        printf("%s", (i == declarators.size() - 1) ? "]" : ", ");
    }
}

///////////////////////////////////////////////////////////////////////////

void
GetStructTypesNamesPositions(const std::vector<StructDeclaration *> &sd,
                             std::vector<const Type *> *elementTypes,
                             std::vector<std::string> *elementNames,
                             std::vector<SourcePos> *elementPositions) {
    for (unsigned int i = 0; i < sd.size(); ++i) {
        const Type *type = sd[i]->type;
        // FIXME: making this fake little DeclSpecs here is really
        // disgusting
        DeclSpecs ds(type);
        if (type->IsUniformType()) 
            ds.typeQualifiers |= TYPEQUAL_UNIFORM;
        else
            ds.typeQualifiers |= TYPEQUAL_VARYING;

        for (unsigned int j = 0; j < sd[i]->declarators->size(); ++j) {
            Declarator *d = (*sd[i]->declarators)[j];
            d->InitFromDeclSpecs(&ds);

            // if it's an unsized array, make it a reference to an unsized
            // array, so the caller can pass a pointer...
            Symbol *sym = d->GetSymbol();
            const ArrayType *at = dynamic_cast<const ArrayType *>(sym->type);
            if (at && at->GetElementCount() == 0)
                sym->type = new ReferenceType(sym->type, type->IsConstType());

            elementTypes->push_back(sym->type);
            elementNames->push_back(sym->name);
            elementPositions->push_back(sym->pos);
        }
    }
}
