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
#include "sym.h"
#include "type.h"
#include "expr.h"
#include <stdio.h>

///////////////////////////////////////////////////////////////////////////
// DeclSpecs

DeclSpecs::DeclSpecs(const Type *t, StorageClass sc, int tq) {
    baseType = t;
    storageClass = sc;
    typeQualifier = tq;
    soaWidth = 0;
    vectorSize = 0;
}


void
DeclSpecs::Print() const {
    if (storageClass == SC_EXTERN)   printf("extern ");
    if (storageClass == SC_EXTERN_C) printf("extern \"C\" ");
    if (storageClass == SC_EXPORT)   printf("export ");
    if (storageClass == SC_STATIC)   printf("static ");
    if (storageClass == SC_TYPEDEF)  printf("typedef ");

    if (soaWidth > 0) printf("soa<%d> ", soaWidth);

    if (typeQualifier & TYPEQUAL_INLINE)    printf("inline ");
    if (typeQualifier & TYPEQUAL_CONST)     printf("const ");
    if (typeQualifier & TYPEQUAL_UNIFORM)   printf("uniform ");
    if (typeQualifier & TYPEQUAL_VARYING)   printf("varying ");
    if (typeQualifier & TYPEQUAL_TASK)      printf("task ");
    if (typeQualifier & TYPEQUAL_REFERENCE) printf("reference ");
    if (typeQualifier & TYPEQUAL_UNSIGNED)  printf("unsigned ");

    printf("%s", baseType->GetString().c_str());

    if (vectorSize > 0) printf("<%d>", vectorSize);
}


///////////////////////////////////////////////////////////////////////////
// Declarator

Declarator::Declarator(Symbol *s, SourcePos p) 
  : pos(p) { 
    sym = s;
    functionArgs = NULL;
    isFunction = false;
    initExpr = NULL;
}


void
Declarator::AddArrayDimension(int size) {
    assert(size > 0 || size == -1); // -1 -> unsized
    arraySize.push_back(size);
}


void
Declarator::InitFromDeclSpecs(DeclSpecs *ds) {
    sym->type = GetType(ds);

    if (ds->storageClass == SC_STATIC)
        sym->isStatic = true;
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


static const Type *
lGetType(const Declarator *decl, DeclSpecs *ds, 
         std::vector<int>::const_iterator arrayIter) {
    if (arrayIter == decl->arraySize.end()) {
        // If we don't have an array (or have processed all of the array
        // dimensions in previous recursive calls), we can go ahead and
        // figure out the final non-array type we have here.
        const Type *type = ds->baseType;
        if (type == NULL) {
            Error(decl->pos, "Type not provided in variable declaration for variable \"%s\".",
                  decl->sym->name.c_str());
            return NULL;
        }

        // Account for 'unsigned' and 'const' qualifiers in the type
        if ((ds->typeQualifier & TYPEQUAL_UNSIGNED) != 0) {
            const Type *unsignedType = type->GetAsUnsignedType();
            if (unsignedType != NULL)
                type = unsignedType;
            else
                Error(decl->pos, "\"unsigned\" qualifier is illegal with \"%s\" type.",
                      type->GetString().c_str());
        }
        if ((ds->typeQualifier & TYPEQUAL_CONST) != 0)
            type = type->GetAsConstType();

        if (ds->vectorSize > 0) {
            const AtomicType *atomicType = dynamic_cast<const AtomicType *>(type);
            if (atomicType == NULL) {
                Error(decl->pos, "Only atomic types (int, float, ...) are legal for vector "
                      "types.");
                return NULL;
            }
            type = new VectorType(atomicType, ds->vectorSize);
        }

        // if uniform/varying is specified explicitly, then go with that
        if ((ds->typeQualifier & TYPEQUAL_UNIFORM) != 0)
            return type->GetAsUniformType();
        else if ((ds->typeQualifier & TYPEQUAL_VARYING) != 0)
            return type->GetAsVaryingType();
        else {
            // otherwise, structs are uniform by default and everything
            // else is varying by default
            if (dynamic_cast<const StructType *>(type) != NULL)
                return type->GetAsUniformType();
            else
                return type->GetAsVaryingType();
        }
    }
    else {
        // Peel off one dimension of the array
        int arraySize = *arrayIter;
        ++arrayIter;

        // Get the type, not including the arraySize dimension peeled off
        // above.
        const Type *childType = lGetType(decl, ds, arrayIter);

        int soaWidth = ds->soaWidth;
        if (soaWidth == 0)
            // If there's no "soa<n>" stuff going on, just return a regular
            // array with the appropriate size 
            return new ArrayType(childType, arraySize == -1 ? 0 : arraySize);
       else {
            // Make sure we actually have an array of structs ..
            const StructType *childStructType = 
                dynamic_cast<const StructType *>(childType);
            if (childStructType == NULL) {
                Error(decl->pos, "Illegal to provide soa<%d> qualifier with non-struct "
                      "type \"%s\".", soaWidth, childType->GetString().c_str());
                return new ArrayType(childType, arraySize == -1 ? 0 : arraySize);
            }
            else if ((soaWidth & (soaWidth - 1)) != 0) {
                Error(decl->pos, "soa<%d> width illegal.  Value must be power of two.",
                      soaWidth);
                return NULL;
            }
            else if (arraySize != -1 && (arraySize % soaWidth) != 0) {
                Error(decl->pos, "soa<%d> width must evenly divide array size %d.",
                      soaWidth, arraySize);
                return NULL;
            }
            return new SOAArrayType(childStructType, arraySize == -1 ? 0 : arraySize,
                                    soaWidth);
        }
    }
}


const Type *
Declarator::GetType(DeclSpecs *ds) const {
    bool hasUniformQual = ((ds->typeQualifier & TYPEQUAL_UNIFORM) != 0);
    bool hasVaryingQual = ((ds->typeQualifier & TYPEQUAL_VARYING) != 0);
    bool isTask =         ((ds->typeQualifier & TYPEQUAL_TASK) != 0);
    bool isReference =    ((ds->typeQualifier & TYPEQUAL_REFERENCE) != 0);

    if (hasUniformQual && hasVaryingQual) {
        Error(pos, "Can't provide both \"uniform\" and \"varying\" qualifiers.");
        return NULL;
    }

    if (isFunction) {
        std::vector<const Type *> args;
        std::vector<std::string> argNames;
        if (functionArgs) {
            // Loop over the function arguments and get names and types for
            // each one in the args and argNames arrays
            for (unsigned int i = 0; i < functionArgs->size(); ++i) {
                Declaration *d = (*functionArgs)[i];
                Symbol *sym;
                if (d->declarators.size() == 0) {
                    // function declaration like foo(float), w/o a name for
                    // the parameter
                    char buf[32];
                    sprintf(buf, "__anon_parameter_%d", i);
                    sym = new Symbol(buf, pos);
                    Declarator *declarator = new Declarator(sym, sym->pos);
                    sym->type = declarator->GetType(ds);
                    d->declarators.push_back(declarator);
                }
                else {
                    assert(d->declarators.size() == 1);
                    sym = d->declarators[0]->sym;
                }

                // Arrays are passed by reference, so convert array
                // parameters to be references here.
                if (dynamic_cast<const ArrayType *>(sym->type) != NULL)
                    sym->type = new ReferenceType(sym->type, sym->type->IsConstType());

                args.push_back(sym->type);
                argNames.push_back(sym->name);
            }
        }

        if (ds->baseType == NULL) {
            Warning(pos, "No return type provided in declaration of function \"%s\". "
                    "Treating as \"void\".", sym->name.c_str());
            ds->baseType = AtomicType::Void;
        }

        if (isReference) {
            Error(pos, "Function return types can't be reference types.");
            return NULL;
        }

        const Type *returnType = lGetType(this, ds, arraySize.begin());
        if (returnType == NULL)
            return NULL;

        bool isExported = (ds->storageClass == SC_EXPORT);
        bool isExternC =  (ds->storageClass == SC_EXTERN_C);
        return new FunctionType(returnType, args, pos, &argNames, isTask, 
                                isExported, isExternC);
    }
    else {
        if (isTask)
            Error(pos, "\"task\" qualifier illegal in variable declaration \"%s\".",
                  sym->name.c_str());

        const Type *type = lGetType(this, ds, arraySize.begin());

        if (type != NULL && isReference) {
            bool hasConstQual = ((ds->typeQualifier & TYPEQUAL_CONST) != 0);
            type = new ReferenceType(type, hasConstQual);
        }

        return type;
    }
}

///////////////////////////////////////////////////////////////////////////
// Declaration

void
Declaration::AddSymbols(SymbolTable *st) const {
    assert(declSpecs->storageClass != SC_TYPEDEF);

    for (unsigned int i = 0; i < declarators.size(); ++i)
       if (declarators[i])
           st->AddVariable(declarators[i]->sym);
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
            ds.typeQualifier |= TYPEQUAL_UNIFORM;
        else
            ds.typeQualifier |= TYPEQUAL_VARYING;

        for (unsigned int j = 0; j < sd[i]->declarators->size(); ++j) {
            Declarator *d = (*sd[i]->declarators)[j];
            d->InitFromDeclSpecs(&ds);

            // if it's an unsized array, make it a reference to an unsized
            // array, so the caller can pass a pointer...
            const ArrayType *at = dynamic_cast<const ArrayType *>(d->sym->type);
            if (at && at->GetElementCount() == 0)
                d->sym->type = new ReferenceType(d->sym->type, type->IsConstType());

            elementTypes->push_back(d->sym->type);
            elementNames->push_back(d->sym->name);
            elementPositions->push_back(d->sym->pos);
        }
    }
}
