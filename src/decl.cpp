/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file decl.cpp
    @brief Implementations of classes related to turning declarations into
           symbol names and types.
*/

#include "decl.h"
#include "expr.h"
#include "module.h"
#include "stmt.h"
#include "sym.h"
#include "type.h"
#include "util.h"

#include <set>
#include <stdio.h>
#include <string.h>

using namespace ispc;

static void lPrintTypeQualifiers(int typeQualifiers) {
    if (typeQualifiers & TYPEQUAL_INLINE)
        printf("inline ");
    if (typeQualifiers & TYPEQUAL_CONST)
        printf("const ");
    if (typeQualifiers & TYPEQUAL_UNIFORM)
        printf("uniform ");
    if (typeQualifiers & TYPEQUAL_VARYING)
        printf("varying ");
    if (typeQualifiers & TYPEQUAL_TASK)
        printf("task ");
    if (typeQualifiers & TYPEQUAL_SIGNED)
        printf("signed ");
    if (typeQualifiers & TYPEQUAL_UNSIGNED)
        printf("unsigned ");
    if (typeQualifiers & TYPEQUAL_EXPORT)
        printf("export ");
    if (typeQualifiers & TYPEQUAL_UNMASKED)
        printf("unmasked ");
}

/** Given a Type and a set of type qualifiers, apply the type qualifiers to
    the type, returning the type that is the result.
*/
static const Type *lApplyTypeQualifiers(int typeQualifiers, const Type *type, SourcePos pos) {
    if (type == nullptr)
        return nullptr;

    if ((typeQualifiers & TYPEQUAL_CONST) != 0) {
        type = type->GetAsConstType();
    }

    if (((typeQualifiers & TYPEQUAL_UNIFORM) != 0) && ((typeQualifiers & TYPEQUAL_VARYING) != 0)) {
        Error(pos, "Type \"%s\" cannot be qualified with both uniform and varying.", type->GetString().c_str());
    }

    if ((typeQualifiers & TYPEQUAL_UNIFORM) != 0) {
        if (type->IsVoidType())
            Error(pos, "\"uniform\" qualifier is illegal with \"void\" type.");
        else
            type = type->GetAsUniformType();
    } else if ((typeQualifiers & TYPEQUAL_VARYING) != 0) {
        if (type->IsVoidType())
            Error(pos, "\"varying\" qualifier is illegal with \"void\" type.");
        else
            type = type->GetAsVaryingType();
    } else {
        if (type->IsVoidType() == false)
            type = type->GetAsUnboundVariabilityType();
    }

    if ((typeQualifiers & TYPEQUAL_UNSIGNED) != 0) {
        if ((typeQualifiers & TYPEQUAL_SIGNED) != 0)
            Error(pos, "Illegal to apply both \"signed\" and \"unsigned\" "
                       "qualifiers.");

        const Type *unsignedType = type->GetAsUnsignedType();
        if (unsignedType != nullptr)
            type = unsignedType;
        else {
            const Type *resolvedType = type->ResolveUnboundVariability(Variability::Varying);
            Error(pos, "\"unsigned\" qualifier is illegal with \"%s\" type.", resolvedType->GetString().c_str());
        }
    }

    if ((typeQualifiers & TYPEQUAL_SIGNED) != 0 && type->IsIntType() == false) {
        const Type *resolvedType = type->ResolveUnboundVariability(Variability::Varying);
        Error(pos,
              "\"signed\" qualifier is illegal with non-integer type "
              "\"%s\".",
              resolvedType->GetString().c_str());
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
    if (t != nullptr) {
        if (m->symbolTable->ContainsType(t)) {
            // Typedefs might have uniform/varying qualifiers inside.
            if (t->IsVaryingType()) {
                typeQualifiers |= TYPEQUAL_VARYING;
            } else if (t->IsUniformType()) {
                typeQualifiers |= TYPEQUAL_UNIFORM;
            }
        }
    }
}

const Type *DeclSpecs::GetBaseType(SourcePos pos) const {
    const Type *retType = baseType;

    if (retType == nullptr) {
        Warning(pos, "No type specified in declaration.  Assuming int32.");
        retType = AtomicType::UniformInt32->GetAsUnboundVariabilityType();
    }

    if (vectorSize > 0) {
        const AtomicType *atomicType = CastType<AtomicType>(retType);
        if (atomicType == nullptr) {
            Error(pos, "Only atomic types (int, float, ...) are legal for vector "
                       "types.");
            return nullptr;
        }
        retType = new VectorType(atomicType, vectorSize);
    }

    retType = lApplyTypeQualifiers(typeQualifiers, retType, pos);

    if (soaWidth > 0) {
        const StructType *st = CastType<StructType>(retType);

        if (st == nullptr) {
            Error(pos,
                  "Illegal to provide soa<%d> qualifier with non-struct "
                  "type \"%s\".",
                  soaWidth, retType ? retType->GetString().c_str() : "NULL");
            return nullptr;
        } else if (soaWidth <= 0 || (soaWidth & (soaWidth - 1)) != 0) {
            Error(pos,
                  "soa<%d> width illegal. Value must be positive power "
                  "of two.",
                  soaWidth);
            return nullptr;
        }

        if (st->IsUniformType()) {
            Error(pos,
                  "\"uniform\" qualifier and \"soa<%d>\" qualifier can't "
                  "both be used in a type declaration.",
                  soaWidth);
            return nullptr;
        } else if (st->IsVaryingType()) {
            Error(pos,
                  "\"varying\" qualifier and \"soa<%d>\" qualifier can't "
                  "both be used in a type declaration.",
                  soaWidth);
            return nullptr;
        } else
            retType = st->GetAsSOAType(soaWidth);

        if (soaWidth < g->target->getVectorWidth())
            PerformanceWarning(pos,
                               "soa<%d> width smaller than gang size %d "
                               "currently leads to inefficient code to access "
                               "soa types.",
                               soaWidth, g->target->getVectorWidth());
    }

    return retType;
}

static const char *lGetStorageClassName(StorageClass storageClass) {
    switch (storageClass) {
    case SC_NONE:
        return "";
    case SC_EXTERN:
        return "extern";
    case SC_EXTERN_C:
        return "extern \"C\"";
    case SC_EXTERN_SYCL:
        return "extern \"SYCL\"";
    case SC_STATIC:
        return "static";
    case SC_TYPEDEF:
        return "typedef";
    default:
        FATAL("Unhandled storage class in lGetStorageClassName");
        return "";
    }
}

void DeclSpecs::Print() const {
    printf("Declspecs: [%s ", lGetStorageClassName(storageClass));

    if (soaWidth > 0)
        printf("soa<%d> ", soaWidth);
    lPrintTypeQualifiers(typeQualifiers);
    printf("base type: %s", baseType->GetString().c_str());

    if (vectorSize > 0)
        printf("<%d>", vectorSize);
    printf("]");
}

///////////////////////////////////////////////////////////////////////////
// Declarator

Declarator::Declarator(DeclaratorKind dk, SourcePos p) : pos(p), kind(dk) {
    child = nullptr;
    typeQualifiers = 0;
    storageClass = SC_NONE;
    arraySize = -1;
    type = nullptr;
    initExpr = nullptr;
}

void Declarator::InitFromDeclSpecs(DeclSpecs *ds) {
    const Type *baseType = ds->GetBaseType(pos);
    if (!baseType) {
        AssertPos(pos, m->errorCount > 0);
        return;
    }

    InitFromType(baseType, ds);

    if (type == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return;
    }

    storageClass = ds->storageClass;

    if (ds->declSpecList.size() > 0 && CastType<FunctionType>(type) == nullptr) {
        Error(pos,
              "__declspec specifiers for non-function type \"%s\" are "
              "not used.",
              type->GetString().c_str());
    }
}

void Declarator::Print() const {
    Indent indent;
    indent.pushSingle();
    Print(indent);
    fflush(stdout);
}

void Declarator::Print(Indent &indent) const {
    indent.Print("Declarator", pos);

    printf("[");
    lPrintTypeQualifiers(typeQualifiers);
    printf("%s ", lGetStorageClassName(storageClass));
    if (name.size() > 0)
        printf("%s", name.c_str());
    else
        printf("(unnamed)");

    printf(", array size = %d", arraySize);

    printf(", kind = ");
    switch (kind) {
    case DK_BASE:
        printf("base");
        break;
    case DK_POINTER:
        printf("pointer");
        break;
    case DK_REFERENCE:
        printf("reference");
        break;
    case DK_ARRAY:
        printf("array");
        break;
    case DK_FUNCTION:
        printf("function");
        break;
    default:
        FATAL("Unhandled declarator kind");
    }

    printf("]\n");

    int kids = (initExpr ? 1 : 0) + functionParams.size() + (child ? 1 : 0);
    indent.pushList(kids);

    if (initExpr != nullptr) {
        indent.setNextLabel("init");
        initExpr->Print(indent);
    }

    if (functionParams.size() > 0) {
        for (unsigned int i = 0; i < functionParams.size(); ++i) {
            static constexpr std::size_t BUFSIZE{20};
            char buffer[BUFSIZE];
            snprintf(buffer, BUFSIZE, "func param %d", i);
            indent.setNextLabel(buffer);
            functionParams[i]->Print(indent);
        }
    }

    if (child != nullptr) {
        indent.setNextLabel("child");
        child->Print(indent);
    }

    indent.Done();
}

void Declarator::InitFromType(const Type *baseType, DeclSpecs *ds) {
    bool hasUniformQual = ((typeQualifiers & TYPEQUAL_UNIFORM) != 0);
    bool hasVaryingQual = ((typeQualifiers & TYPEQUAL_VARYING) != 0);
    bool isTask = ((typeQualifiers & TYPEQUAL_TASK) != 0);
    bool isExported = ((typeQualifiers & TYPEQUAL_EXPORT) != 0);
    bool isConst = ((typeQualifiers & TYPEQUAL_CONST) != 0);
    bool isUnmasked = ((typeQualifiers & TYPEQUAL_UNMASKED) != 0);

    if (hasUniformQual && hasVaryingQual) {
        Error(pos, "Can't provide both \"uniform\" and \"varying\" qualifiers.");
        return;
    }
    if (kind != DK_FUNCTION && isTask) {
        Error(pos, "\"task\" qualifier illegal in variable declaration.");
        return;
    }
    if (kind != DK_FUNCTION && isUnmasked) {
        Error(pos, "\"unmasked\" qualifier illegal in variable declaration.");
        return;
    }
    if (kind != DK_FUNCTION && isExported) {
        Error(pos, "\"export\" qualifier illegal in variable declaration.");
        return;
    }

    Variability variability(Variability::Unbound);
    if (hasUniformQual)
        variability = Variability::Uniform;
    else if (hasVaryingQual)
        variability = Variability::Varying;

    if (kind == DK_BASE) {
        // All of the type qualifiers should be in the DeclSpecs for the
        // base declarator
        AssertPos(pos, typeQualifiers == 0);
        AssertPos(pos, child == nullptr);
        type = baseType;
    } else if (kind == DK_POINTER) {
        /* For now, any pointer to an SOA type gets the slice property; if
           we add the capability to declare pointers as slices or not,
           we'll want to set this based on a type qualifier here. */
        const Type *ptrType = new PointerType(baseType, variability, isConst, baseType->IsSOAType());
        if (child != nullptr) {
            child->InitFromType(ptrType, ds);
            type = child->type;
            name = child->name;
        } else
            type = ptrType;
    } else if (kind == DK_REFERENCE) {
        if (hasUniformQual) {
            Error(pos, "\"uniform\" qualifier is illegal to apply to references.");
            return;
        }
        if (hasVaryingQual) {
            Error(pos, "\"varying\" qualifier is illegal to apply to references.");
            return;
        }
        if (isConst) {
            Error(pos, "\"const\" qualifier is to illegal apply to references.");
            return;
        }
        // The parser should disallow this already, but double check.
        if (CastType<ReferenceType>(baseType) != nullptr) {
            Error(pos, "References to references are illegal.");
            return;
        }

        const Type *refType = new ReferenceType(baseType);
        if (child != nullptr) {
            child->InitFromType(refType, ds);
            type = child->type;
            name = child->name;
        } else
            type = refType;
    } else if (kind == DK_ARRAY) {
        if (baseType->IsVoidType()) {
            Error(pos, "Arrays of \"void\" type are illegal.");
            return;
        }
        if (CastType<ReferenceType>(baseType)) {
            Error(pos, "Arrays of references (type \"%s\") are illegal.", baseType->GetString().c_str());
            return;
        }

        const Type *arrayType = new ArrayType(baseType, arraySize);
        if (child != nullptr) {
            child->InitFromType(arrayType, ds);
            type = child->type;
            name = child->name;
        } else {
            type = arrayType;
        }
    } else if (kind == DK_FUNCTION) {
        llvm::SmallVector<const Type *, 8> args;
        llvm::SmallVector<std::string, 8> argNames;
        llvm::SmallVector<Expr *, 8> argDefaults;
        llvm::SmallVector<SourcePos, 8> argPos;

        // Loop over the function arguments and store the names, types,
        // default values (if any), and source file positions each one in
        // the corresponding vector.
        for (unsigned int i = 0; i < functionParams.size(); ++i) {
            Declaration *d = functionParams[i];

            if (d == nullptr) {
                AssertPos(pos, m->errorCount > 0);
                continue;
            }
            if (d->declarators.size() == 0) {
                // function declaration like foo(float), w/o a name for the
                // parameter; wire up a placeholder Declarator for it
                d->declarators.push_back(new Declarator(DK_BASE, pos));
                d->declarators[0]->InitFromDeclSpecs(d->declSpecs);
            }

            AssertPos(pos, d->declarators.size() == 1);
            Declarator *decl = d->declarators[0];
            if (decl == nullptr || decl->type == nullptr) {
                AssertPos(pos, m->errorCount > 0);
                continue;
            }

            if (decl->name == "") {
                // Give a name to any anonymous parameter declarations
                char buf[32];
                snprintf(buf, sizeof(buf), "__anon_parameter_%d", i);
                decl->name = buf;
            }
            if (!decl->type->IsDependentType()) {
                decl->type = decl->type->ResolveUnboundVariability(Variability::Varying);
            }

            if (d->declSpecs->storageClass != SC_NONE)
                Error(decl->pos,
                      "Storage class \"%s\" is illegal in "
                      "function parameter declaration for parameter \"%s\".",
                      lGetStorageClassName(d->declSpecs->storageClass), decl->name.c_str());
            if (decl->type->IsVoidType()) {
                Error(decl->pos, "Parameter with type \"void\" illegal in function "
                                 "parameter list.");
                decl->type = nullptr;
            }

            const ArrayType *at = CastType<ArrayType>(decl->type);
            if (at != nullptr) {
                // As in C, arrays are passed to functions as pointers to
                // their element type.  We'll just immediately make this
                // change now.  (One shortcoming of losing the fact that
                // the it was originally an array is that any warnings or
                // errors later issued that print the function type will
                // report this differently than it was originally declared
                // in the function, but it's not clear that this is a
                // significant problem.)
                const Type *targetType = at->GetElementType();
                if (targetType == nullptr) {
                    AssertPos(pos, m->errorCount > 0);
                    return;
                }

                decl->type = PointerType::GetUniform(targetType, at->IsSOAType());

                // Make sure there are no unsized arrays (other than the
                // first dimension) in function parameter lists.
                at = CastType<ArrayType>(targetType);
                while (at != nullptr) {
                    if (at->GetElementCount() == 0)
                        Error(decl->pos, "Arrays with unsized dimensions in "
                                         "dimensions after the first one are illegal in "
                                         "function parameter lists.");
                    at = CastType<ArrayType>(at->GetElementType());
                }
            }

            args.push_back(decl->type);
            argNames.push_back(decl->name);
            argPos.push_back(decl->pos);

            Expr *init = nullptr;
            // Try to find an initializer expression.
            while (decl != nullptr) {
                if (decl->initExpr != nullptr) {
                    decl->initExpr = TypeCheck(decl->initExpr);
                    decl->initExpr = Optimize(decl->initExpr);
                    if (decl->initExpr != nullptr) {
                        init = llvm::dyn_cast<ConstExpr>(decl->initExpr);
                        if (init == nullptr)
                            init = llvm::dyn_cast<NullPointerExpr>(decl->initExpr);
                        if (init == nullptr)
                            Error(decl->initExpr->pos,
                                  "Default value for parameter "
                                  "\"%s\" must be a compile-time constant.",
                                  decl->name.c_str());
                    }
                    break;
                } else
                    decl = decl->child;
            }
            argDefaults.push_back(init);
        }

        const Type *returnType = baseType;
        if (returnType == nullptr) {
            Error(pos, "No return type provided in function declaration.");
            return;
        }

        if (CastType<FunctionType>(returnType) != nullptr) {
            Error(pos, "Illegal to return function type from function.");
            return;
        }

        if (!returnType->IsDependentType()) {
            returnType = returnType->ResolveUnboundVariability(Variability::Varying);
        }

        bool isExternC = ds && (ds->storageClass == SC_EXTERN_C);
        bool isExternSYCL = ds && (ds->storageClass == SC_EXTERN_SYCL);
        bool isExported = ds && ((ds->typeQualifiers & TYPEQUAL_EXPORT) != 0);
        bool isTask = ds && ((ds->typeQualifiers & TYPEQUAL_TASK) != 0);
        bool isUnmasked = ds && ((ds->typeQualifiers & TYPEQUAL_UNMASKED) != 0);
        bool isVectorCall = ds && ((ds->typeQualifiers & TYPEQUAL_VECTORCALL) != 0);
        bool isRegCall = ds && ((ds->typeQualifiers & TYPEQUAL_REGCALL) != 0);

        if (isExported && isTask) {
            Error(pos, "Function can't have both \"task\" and \"export\" "
                       "qualifiers");
            return;
        }
        if (isExternC && isTask) {
            Error(pos, "Function can't have both \"extern \"C\"\" and \"task\" "
                       "qualifiers");
            return;
        }
        if (isExternC && isExported) {
            Error(pos, "Function can't have both \"extern \"C\"\" and \"export\" "
                       "qualifiers");
            return;
        }
        if (isExternSYCL && isTask) {
            Error(pos, "Function can't have both \"extern \"SYCL\"\" and \"task\" "
                       "qualifiers");
            return;
        }
        if (isExternSYCL && isExported) {
            Error(pos, "Function can't have both \"extern \"SYCL\"\" and \"export\" "
                       "qualifiers");
            return;
        }
        if (isUnmasked && isExported)
            Warning(pos, "\"unmasked\" qualifier is redundant for exported "
                         "functions.");

        if (child == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return;
        }

        const FunctionType *functionType =
            new FunctionType(returnType, args, argNames, argDefaults, argPos, isTask, isExported, isExternC,
                             isExternSYCL, isUnmasked, isVectorCall, isRegCall);

        // handle any explicit __declspecs on the function
        if (ds != nullptr) {
            for (int i = 0; i < (int)ds->declSpecList.size(); ++i) {
                std::string str = ds->declSpecList[i].first;
                SourcePos ds_spec_pos = ds->declSpecList[i].second;

                if (str == "safe")
                    (const_cast<FunctionType *>(functionType))->isSafe = true;
                else if (!strncmp(str.c_str(), "cost", 4)) {
                    int cost = atoi(str.c_str() + 4);
                    if (cost < 0)
                        Error(ds_spec_pos, "Negative function cost %d is illegal.", cost);
                    (const_cast<FunctionType *>(functionType))->costOverride = cost;
                } else
                    Error(ds_spec_pos, "__declspec parameter \"%s\" unknown.", str.c_str());
            }
        }

        child->InitFromType(functionType, ds);
        type = child->type;
        name = child->name;
    } else {
        UNREACHABLE();
    }
}

///////////////////////////////////////////////////////////////////////////
// Declaration

Declaration::Declaration(DeclSpecs *ds, std::vector<Declarator *> *dlist) {
    declSpecs = ds;
    if (dlist != nullptr)
        declarators = *dlist;
    for (unsigned int i = 0; i < declarators.size(); ++i)
        if (declarators[i] != nullptr)
            declarators[i]->InitFromDeclSpecs(declSpecs);
}

Declaration::Declaration(DeclSpecs *ds, Declarator *d) {
    declSpecs = ds;
    if (d != nullptr) {
        d->InitFromDeclSpecs(ds);
        declarators.push_back(d);
    }
}

std::vector<VariableDeclaration> Declaration::GetVariableDeclarations() const {
    Assert(declSpecs->storageClass != SC_TYPEDEF);
    std::vector<VariableDeclaration> vars;

    for (unsigned int i = 0; i < declarators.size(); ++i) {
        Declarator *decl = declarators[i];
        if (decl == nullptr || decl->type == nullptr) {
            // Ignore earlier errors
            Assert(m->errorCount > 0);
            continue;
        }

        if (decl->type->IsVoidType())
            Error(decl->pos, "\"void\" type variable illegal in declaration.");
        else if (CastType<FunctionType>(decl->type) == nullptr) {
            if (!decl->type->IsDependentType()) {
                decl->type = decl->type->ResolveUnboundVariability(Variability::Varying);
            }
            Symbol *sym = new Symbol(decl->name, decl->pos, decl->type, decl->storageClass);
            m->symbolTable->AddVariable(sym);
            vars.push_back(VariableDeclaration(sym, decl->initExpr));
        } else {
            Error(decl->pos, "\"%s\" is illegal in declaration.", decl->name.c_str());
        }
    }

    return vars;
}

void Declaration::DeclareFunctions() {
    Assert(declSpecs->storageClass != SC_TYPEDEF);

    for (unsigned int i = 0; i < declarators.size(); ++i) {
        Declarator *decl = declarators[i];
        if (decl == nullptr || decl->type == nullptr) {
            // Ignore earlier errors
            Assert(m->errorCount > 0);
            continue;
        }

        const FunctionType *ftype = CastType<FunctionType>(decl->type);
        if (ftype == nullptr)
            continue;

        bool isInline = (declSpecs->typeQualifiers & TYPEQUAL_INLINE);
        bool isNoInline = (declSpecs->typeQualifiers & TYPEQUAL_NOINLINE);
        bool isVectorCall = (declSpecs->typeQualifiers & TYPEQUAL_VECTORCALL);
        bool isRegCall = (declSpecs->typeQualifiers & TYPEQUAL_REGCALL);
        m->AddFunctionDeclaration(decl->name, ftype, decl->storageClass, isInline, isNoInline, isVectorCall, isRegCall,
                                  decl->pos);
    }
}

void Declaration::Print() const {
    Indent indent;
    indent.pushSingle();
    Print(indent);
    fflush(stdout);
}

void Declaration::Print(Indent &indent) const {
    indent.Print("Declaration: specs [");
    declSpecs->Print();
    printf("], declarators:\n");

    indent.pushList(declarators.size());
    for (unsigned int i = 0; i < declarators.size(); ++i) {
        declarators[i]->Print(indent);
    }

    indent.Done();
}

///////////////////////////////////////////////////////////////////////////

void ispc::GetStructTypesNamesPositions(const std::vector<StructDeclaration *> &sd,
                                        llvm::SmallVector<const Type *, 8> *elementTypes,
                                        llvm::SmallVector<std::string, 8> *elementNames,
                                        llvm::SmallVector<SourcePos, 8> *elementPositions) {
    std::set<std::string> seenNames;
    for (unsigned int i = 0; i < sd.size(); ++i) {
        const Type *type = sd[i]->type;
        if (type == nullptr)
            continue;

        // FIXME: making this fake little DeclSpecs here is really
        // disgusting
        DeclSpecs ds(type);
        if (type->IsVoidType() == false) {
            if (type->IsUniformType())
                ds.typeQualifiers |= TYPEQUAL_UNIFORM;
            else if (type->IsVaryingType())
                ds.typeQualifiers |= TYPEQUAL_VARYING;
            else if (type->GetSOAWidth() != 0)
                ds.soaWidth = type->GetSOAWidth();
            // FIXME: ds.vectorSize?
        }

        for (unsigned int j = 0; j < sd[i]->declarators->size(); ++j) {
            Declarator *d = (*sd[i]->declarators)[j];
            d->InitFromDeclSpecs(&ds);

            if (d->type->IsVoidType())
                Error(d->pos, "\"void\" type illegal for struct member.");

            elementTypes->push_back(d->type);

            if (seenNames.find(d->name) != seenNames.end())
                Error(d->pos,
                      "Struct member \"%s\" has same name as a "
                      "previously-declared member.",
                      d->name.c_str());
            else
                seenNames.insert(d->name);

            elementNames->push_back(d->name);
            elementPositions->push_back(d->pos);
        }
    }

    for (int i = 0; i < (int)elementTypes->size() - 1; ++i) {
        const ArrayType *arrayType = CastType<ArrayType>((*elementTypes)[i]);

        if (arrayType != nullptr && arrayType->GetElementCount() == 0)
            Error((*elementPositions)[i], "Unsized arrays aren't allowed except "
                                          "for the last member in a struct definition.");
    }
}
