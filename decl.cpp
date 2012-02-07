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
#include <set>

static void
lPrintTypeQualifiers(int typeQualifiers) {
    if (typeQualifiers & TYPEQUAL_INLINE)    printf("inline ");
    if (typeQualifiers & TYPEQUAL_CONST)     printf("const ");
    if (typeQualifiers & TYPEQUAL_UNIFORM)   printf("uniform ");
    if (typeQualifiers & TYPEQUAL_VARYING)   printf("varying ");
    if (typeQualifiers & TYPEQUAL_TASK)      printf("task ");
    if (typeQualifiers & TYPEQUAL_SIGNED)    printf("signed ");
    if (typeQualifiers & TYPEQUAL_UNSIGNED)  printf("unsigned ");
}


/** Given a Type and a set of type qualifiers, apply the type qualifiers to
    the type, returning the type that is the result. 
*/
static const Type *
lApplyTypeQualifiers(int typeQualifiers, const Type *type, SourcePos pos) {
    if (type == NULL)
        return NULL;

    if ((typeQualifiers & TYPEQUAL_CONST) != 0)
        type = type->GetAsConstType();

    if ((typeQualifiers & TYPEQUAL_UNIFORM) != 0)
        type = type->GetAsUniformType();
    else if ((typeQualifiers & TYPEQUAL_VARYING) != 0)
        type = type->GetAsVaryingType();
    else
        type = type->GetAsUnboundVariabilityType();

    if ((typeQualifiers & TYPEQUAL_UNSIGNED) != 0) {
        if ((typeQualifiers & TYPEQUAL_SIGNED) != 0)
            Error(pos, "Illegal to apply both \"signed\" and \"unsigned\" "
                  "qualifiers.");

        const Type *unsignedType = type->GetAsUnsignedType();
        if (unsignedType != NULL)
            type = unsignedType;
        else
            Error(pos, "\"unsigned\" qualifier is illegal with \"%s\" type.",
                  type->ResolveUnboundVariability(Type::Varying)->GetString().c_str());
    }

    if ((typeQualifiers & TYPEQUAL_SIGNED) != 0 && type->IsIntType() == false)
        Error(pos, "\"signed\" qualifier is illegal with non-integer type "
              "\"%s\".", 
              type->ResolveUnboundVariability(Type::Varying)->GetString().c_str());

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

    if (bt == NULL) {
        Warning(pos, "No type specified in declaration.  Assuming int32.");
        bt = AtomicType::UnboundInt32;
    }

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


static const char *
lGetStorageClassName(StorageClass storageClass) {
    switch (storageClass) {
    case SC_NONE:     return "";
    case SC_EXTERN:   return "extern";
    case SC_EXTERN_C: return "extern \"C\"";
    case SC_EXPORT:   return "export";
    case SC_STATIC:   return "static";
    case SC_TYPEDEF:  return "typedef";
    default:          FATAL("Unhandled storage class in lGetStorageClassName");
                      return "";
    }
}


void
DeclSpecs::Print() const {
    printf("Declspecs: [%s ", lGetStorageClassName(storageClass));

    if (soaWidth > 0) printf("soa<%d> ", soaWidth);
    lPrintTypeQualifiers(typeQualifiers);
    printf("base type: %s", baseType->GetString().c_str());

    if (vectorSize > 0) printf("<%d>", vectorSize);
    printf("]");
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
    if (t == NULL) {
        Assert(m->errorCount > 0);
        return;
    }

    Symbol *sym = GetSymbol();
    if (sym != NULL) {
        sym->type = t;
        sym->storageClass = ds->storageClass;
    }
}


Symbol *
Declarator::GetSymbol() const {
    // The symbol lives at the last child in the chain, so walk down there
    // and return the one there.
    const Declarator *d = this;
    while (d->child != NULL)
        d = d->child;
    return d->sym;
}


void
Declarator::Print(int indent) const {
    printf("%*cdeclarator: [", indent, ' ');
    pos.Print();

    lPrintTypeQualifiers(typeQualifiers);
    Symbol *sym = GetSymbol();
    if (sym != NULL)
        printf("%s", sym->name.c_str());
    else
        printf("(null symbol)");

    printf(", array size = %d", arraySize);

    printf(", kind = ");
    switch (kind) {
    case DK_BASE:      printf("base");      break;
    case DK_POINTER:   printf("pointer");   break;
    case DK_REFERENCE: printf("reference"); break;
    case DK_ARRAY:     printf("array");     break;
    case DK_FUNCTION:  printf("function");  break;
    default:           FATAL("Unhandled declarator kind");
    }

    if (initExpr != NULL) {
        printf(" = (");
        initExpr->Print();
        printf(")");
    }

    if (functionParams.size() > 0) {
        for (unsigned int i = 0; i < functionParams.size(); ++i) {
            printf("\n%*cfunc param %d:\n", indent, ' ', i);
            functionParams[i]->Print(indent+4);
        }
    }

    if (child != NULL)
        child->Print(indent + 4);

    printf("]\n");
}


Symbol *
Declarator::GetFunctionInfo(DeclSpecs *ds, std::vector<Symbol *> *funArgs) {
    const FunctionType *type = 
        dynamic_cast<const FunctionType *>(GetType(ds));
    if (type == NULL)
        return NULL;

    Symbol *declSym = GetSymbol();
    Assert(declSym != NULL);

    // Get the symbol for the function from the symbol table.  (It should
    // already have been added to the symbol table by AddGlobal() by the
    // time we get here.)
    Symbol *funSym = m->symbolTable->LookupFunction(declSym->name.c_str(), type);
    if (funSym == NULL)
        // May be NULL due to error earlier in compilation
        Assert(m->errorCount > 0);
    else
        funSym->pos = pos;

    // Walk down to the declarator for the function.  (We have to get past
    // the stuff that specifies the function's return type before we get to
    // the function's declarator.)
    Declarator *d = this;
    while (d != NULL && d->kind != DK_FUNCTION)
        d = d->child;
    Assert(d != NULL);

    for (unsigned int i = 0; i < d->functionParams.size(); ++i) {
        Symbol *sym = d->GetSymbolForFunctionParameter(i);
        if (sym->type == NULL) {
            Assert(m->errorCount > 0);
            continue;
        }
        else
            sym->type = sym->type->ResolveUnboundVariability(Type::Varying);

        funArgs->push_back(sym);
    }

    if (funSym != NULL)
        funSym->type = funSym->type->ResolveUnboundVariability(Type::Varying);

    return funSym;
}


const Type *
Declarator::GetType(const Type *base, DeclSpecs *ds) const {
    bool hasUniformQual = ((typeQualifiers & TYPEQUAL_UNIFORM) != 0);
    bool hasVaryingQual = ((typeQualifiers & TYPEQUAL_VARYING) != 0);
    bool isTask =         ((typeQualifiers & TYPEQUAL_TASK) != 0);
    bool isConst =        ((typeQualifiers & TYPEQUAL_CONST) != 0);

    if (hasUniformQual && hasVaryingQual) {
        Error(pos, "Can't provide both \"uniform\" and \"varying\" qualifiers.");
        return NULL;
    }
    if (kind != DK_FUNCTION && isTask)
        Error(pos, "\"task\" qualifier illegal in variable declaration.");

    Type::Variability variability = Type::Unbound;
    if (hasUniformQual)
        variability = Type::Uniform;
    else if (hasVaryingQual)
        variability = Type::Varying;

    const Type *type = base;
    switch (kind) {
    case DK_BASE:
        // All of the type qualifiers should be in the DeclSpecs for the
        // base declarator
        Assert(typeQualifiers == 0);
        Assert(child == NULL);
        return type;

    case DK_POINTER:
        type = new PointerType(type, variability, isConst);
        if (child != NULL)
            return child->GetType(type, ds);
        else
            return type;
        break;

    case DK_REFERENCE:
        if (hasUniformQual)
            Error(pos, "\"uniform\" qualifier is illegal to apply to references.");
        if (hasVaryingQual)
            Error(pos, "\"varying\" qualifier is illegal to apply to references.");
        if (isConst)
            Error(pos, "\"const\" qualifier is to illegal apply to references.");

        // The parser should disallow this already, but double check.
        if (dynamic_cast<const ReferenceType *>(type) != NULL) {
            Error(pos, "References to references are illegal.");
            return NULL;
        }

        type = new ReferenceType(type);
        if (child != NULL)
            return child->GetType(type, ds);
        else
            return type;
        break;

    case DK_ARRAY:
        if (type == AtomicType::Void) {
            Error(pos, "Arrays of \"void\" type are illegal.");
            return NULL;
        }
        if (dynamic_cast<const ReferenceType *>(type)) {
            Error(pos, "Arrays of references (type \"%s\") are illegal.",
                  type->GetString().c_str());
            return NULL;
        }

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

        // Loop over the function arguments and store the names, types,
        // default values (if any), and source file positions each one in
        // the corresponding vector.
        for (unsigned int i = 0; i < functionParams.size(); ++i) {
            Declaration *d = functionParams[i];

            Symbol *sym = GetSymbolForFunctionParameter(i);

            if (d->declSpecs->storageClass != SC_NONE)
                Error(sym->pos, "Storage class \"%s\" is illegal in "
                      "function parameter declaration for parameter \"%s\".", 
                      lGetStorageClassName(d->declSpecs->storageClass),
                      sym->name.c_str());
            if (sym->type == AtomicType::Void) {
                Error(sym->pos, "Parameter with type \"void\" illegal in function "
                      "parameter list.");
                sym->type = NULL;
            }

            const ArrayType *at = dynamic_cast<const ArrayType *>(sym->type);
            if (at != NULL) {
                // As in C, arrays are passed to functions as pointers to
                // their element type.  We'll just immediately make this
                // change now.  (One shortcoming of losing the fact that
                // the it was originally an array is that any warnings or
                // errors later issued that print the function type will
                // report this differently than it was originally declared
                // in the function, but it's not clear that this is a
                // significant problem.)
                if (at->GetElementType() == NULL) {
                    Assert(m->errorCount > 0);
                    return NULL;
                }

                sym->type = PointerType::GetUniform(at->GetElementType());
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
                // Try to find an initializer expression; if there is one,
                // it lives down to the base declarator.
                Declarator *decl = d->declarators[0];
                while (decl->child != NULL) {
                    Assert(decl->initExpr == NULL);
                    decl = decl->child;
                }

                if (decl->initExpr != NULL &&
                    (decl->initExpr = TypeCheck(decl->initExpr)) != NULL &&
                    (decl->initExpr = Optimize(decl->initExpr)) != NULL &&
                    (init = dynamic_cast<ConstExpr *>(decl->initExpr)) == NULL) {
                    Error(decl->initExpr->pos, "Default value for parameter "
                          "\"%s\" must be a compile-time constant.", 
                          sym->name.c_str());
                }
            }
            argDefaults.push_back(init);
        }

        const Type *returnType = type;
        if (returnType == NULL) {
            Error(pos, "No return type provided in function declaration.");
            return NULL;
        }
        if (dynamic_cast<const FunctionType *>(returnType) != NULL) {
            Error(pos, "Illegal to return function type from function.");
            return NULL;
        }
        
        bool isExported = ds && (ds->storageClass == SC_EXPORT);
        bool isExternC =  ds && (ds->storageClass == SC_EXTERN_C);
        bool isTask =     ds && ((ds->typeQualifiers & TYPEQUAL_TASK) != 0);

        if (isExported && isTask) {
            Error(pos, "Function can't have both \"task\" and \"export\" "
                  "qualifiers");
            return NULL;
        }
        if (isExternC && isTask) {
            Error(pos, "Function can't have both \"extern \"C\"\" and \"task\" "
                  "qualifiers");
            return NULL;
        }
        if (isExternC && isExported) {
            Error(pos, "Function can't have both \"extern \"C\"\" and \"export\" "
                  "qualifiers");
            return NULL;
        }

        if (child == NULL) {
            Assert(m->errorCount > 0);
            return NULL;
        }

        const Type *functionType = 
            new FunctionType(returnType, args, argNames, argDefaults,
                             argPos, isTask, isExported, isExternC);
        functionType = functionType->ResolveUnboundVariability(Type::Varying);
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
    return type;
}


Symbol *
Declarator::GetSymbolForFunctionParameter(int paramNum) const {
    Assert(paramNum < (int)functionParams.size());
    Declaration *d = functionParams[paramNum];

    char buf[32];
    Symbol *sym;
    if (d->declarators.size() == 0) {
        // function declaration like foo(float), w/o a name for
        // the parameter
        sprintf(buf, "__anon_parameter_%d", paramNum);
        sym = new Symbol(buf, pos);
        sym->type = d->declSpecs->GetBaseType(pos);
    }
    else {
        Assert(d->declarators.size() == 1);
        sym = d->declarators[0]->GetSymbol();
        if (sym == NULL) {
            // Handle more complex anonymous declarations like
            // float (float **).
            sprintf(buf, "__anon_parameter_%d", paramNum);
            sym = new Symbol(buf, d->declarators[0]->pos);
            sym->type = d->declarators[0]->GetType(d->declSpecs);
        }
    }
    return sym;
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
    if (d != NULL) {
        d->InitFromDeclSpecs(ds);
        declarators.push_back(d);
    }
}


std::vector<VariableDeclaration>
Declaration::GetVariableDeclarations() const {
    Assert(declSpecs->storageClass != SC_TYPEDEF);
    std::vector<VariableDeclaration> vars;

    for (unsigned int i = 0; i < declarators.size(); ++i) {
        Declarator *decl = declarators[i];
        if (decl == NULL) {
            // Ignore earlier errors
            Assert(m->errorCount > 0);
            continue;
        }

        Symbol *sym = decl->GetSymbol();
        if (sym == NULL || sym->type == NULL) {
            // Ignore errors
            Assert(m->errorCount > 0);
            continue;
        }
        sym->type = sym->type->ResolveUnboundVariability(Type::Varying);

        if (sym->type == AtomicType::Void)
            Error(sym->pos, "\"void\" type variable illegal in declaration.");
        else if (dynamic_cast<const FunctionType *>(sym->type) == NULL) {
            m->symbolTable->AddVariable(sym);
            vars.push_back(VariableDeclaration(sym, decl->initExpr));
        }
    }
    return vars;
}


void
Declaration::DeclareFunctions() {
    Assert(declSpecs->storageClass != SC_TYPEDEF);

    for (unsigned int i = 0; i < declarators.size(); ++i) {
        Declarator *decl = declarators[i];
        if (decl == NULL) {
            // Ignore earlier errors
            Assert(m->errorCount > 0);
            continue;
        }

        Symbol *sym = decl->GetSymbol();
        if (sym == NULL || sym->type == NULL) {
            // Ignore errors
            Assert(m->errorCount > 0);
            continue;
        }
        sym->type = sym->type->ResolveUnboundVariability(Type::Varying);

        if (dynamic_cast<const FunctionType *>(sym->type) == NULL)
            continue;

        bool isInline = (declSpecs->typeQualifiers & TYPEQUAL_INLINE);
        m->AddFunctionDeclaration(sym, isInline);
    }
}


void
Declaration::Print(int indent) const {
    printf("%*cDeclaration: specs [", indent, ' ');
    declSpecs->Print();
    printf("], declarators:\n");
    for (unsigned int i = 0 ; i < declarators.size(); ++i)
        declarators[i]->Print(indent+4);
}

///////////////////////////////////////////////////////////////////////////

void
GetStructTypesNamesPositions(const std::vector<StructDeclaration *> &sd,
                             std::vector<const Type *> *elementTypes,
                             std::vector<std::string> *elementNames,
                             std::vector<SourcePos> *elementPositions) {
    std::set<std::string> seenNames;
    for (unsigned int i = 0; i < sd.size(); ++i) {
        const Type *type = sd[i]->type;
        if (type == NULL)
            continue;

        // FIXME: making this fake little DeclSpecs here is really
        // disgusting
        DeclSpecs ds(type);
        if (type->IsUniformType()) 
            ds.typeQualifiers |= TYPEQUAL_UNIFORM;
        else if (type->IsVaryingType())
            ds.typeQualifiers |= TYPEQUAL_VARYING;

        for (unsigned int j = 0; j < sd[i]->declarators->size(); ++j) {
            Declarator *d = (*sd[i]->declarators)[j];
            d->InitFromDeclSpecs(&ds);

            Symbol *sym = d->GetSymbol();

            if (sym->type == AtomicType::Void)
                Error(d->pos, "\"void\" type illegal for struct member.");

            const ArrayType *arrayType = 
                dynamic_cast<const ArrayType *>(sym->type);
            if (arrayType != NULL && arrayType->GetElementCount() == 0) {
                Error(d->pos, "Unsized arrays aren't allowed in struct "
                      "definitions.");
                elementTypes->push_back(NULL);
            }
            else
                elementTypes->push_back(sym->type);

            if (seenNames.find(sym->name) != seenNames.end())
                Error(d->pos, "Struct member \"%s\" has same name as a "
                      "previously-declared member.", sym->name.c_str());
            else
                seenNames.insert(sym->name);

            elementNames->push_back(sym->name);
            elementPositions->push_back(sym->pos);
        }
    }
}
