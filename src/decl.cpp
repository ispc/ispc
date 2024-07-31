/*
  Copyright (c) 2010-2025, Intel Corporation

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

#include <stdio.h>
#include <string.h>
#include <unordered_set>

using namespace ispc;

void lCheckAddressSpace(int64_t &addrSpace, const std::string &name, SourcePos pos) {
    if (addrSpace < 0) {
        Error(pos, "\"address_space\" attribute must be non-negative, \"%s\".", name.c_str());
        addrSpace = 0;
    }
    if (addrSpace > (int64_t)AddressSpace::ispc_generic) {
        Error(pos, "\"address_space\" attribute %" PRId64 " is out of scope of supported [%d, %d], \"%s\".", addrSpace,
              (int)AddressSpace::ispc_default, (int)AddressSpace::ispc_generic, name.c_str());
        addrSpace = 0;
    }
}

const Type *lGetTypeWithAddressSpace(const Type *type, int addrSpace, const std::string &name, SourcePos pos) {
    if (auto *pt = CastType<PointerType>(type)) {
        type = pt->GetWithAddrSpace((AddressSpace)addrSpace);
    } else if (auto *rt = CastType<ReferenceType>(type)) {
        type = rt->GetWithAddrSpace((AddressSpace)addrSpace);
    } else {
        // ISPC type system seems to support only pointer and reference
        // types with address space, that doesn't look correct in general.
        // Although, it's not a big deal to support it in the future.
        // For now, just issue a warning.
        Warning(pos, "\"address_space\" attribute is only allowed for pointer or reference types, \"%s\".",
                name.c_str());
    }
    return type;
}

void lCheckVariableTypeQualifiers(int typeQualifiers, SourcePos pos) {
    if (typeQualifiers & TYPEQUAL_TASK) {
        Error(pos, "\"task\" qualifier illegal in variable declaration.");
        return;
    }
    if (typeQualifiers & TYPEQUAL_UNMASKED) {
        Error(pos, "\"unmasked\" qualifier illegal in variable declaration.");
        return;
    }
    if (typeQualifiers & TYPEQUAL_EXPORT) {
        Error(pos, "\"export\" qualifier illegal in variable declaration.");
        return;
    }
    if (typeQualifiers & TYPEQUAL_INLINE) {
        Error(pos, "\"inline\" qualifier illegal in variable declaration.");
        return;
    }
    if (typeQualifiers & TYPEQUAL_NOINLINE) {
        Error(pos, "\"noinline\" qualifier illegal in variable declaration.");
        return;
    }
}

void lCheckTypeQualifiers(int typeQualifiers, DeclaratorKind kind, SourcePos pos) {
    if (kind != DK_FUNCTION) {
        lCheckVariableTypeQualifiers(typeQualifiers, pos);
    }
}

bool lIsFunctionKind(Declarator *d) {
    if (!d) {
        return false;
    }

    if (d->kind == DK_FUNCTION) {
        return true;
    }

    if (d->child) {
        return lIsFunctionKind(d->child);
    }

    return false;
}

static void lPrintTypeQualifiers(int typeQualifiers) {
    if (typeQualifiers & TYPEQUAL_INLINE) {
        printf("inline ");
    }
    if (typeQualifiers & TYPEQUAL_CONST) {
        printf("const ");
    }
    if (typeQualifiers & TYPEQUAL_UNIFORM) {
        printf("uniform ");
    }
    if (typeQualifiers & TYPEQUAL_VARYING) {
        printf("varying ");
    }
    if (typeQualifiers & TYPEQUAL_TASK) {
        printf("task ");
    }
    if (typeQualifiers & TYPEQUAL_SIGNED) {
        printf("signed ");
    }
    if (typeQualifiers & TYPEQUAL_UNSIGNED) {
        printf("unsigned ");
    }
    if (typeQualifiers & TYPEQUAL_EXPORT) {
        printf("export ");
    }
    if (typeQualifiers & TYPEQUAL_UNMASKED) {
        printf("unmasked ");
    }
}

/** Given a Type and a set of type qualifiers, apply the type qualifiers to
    the type, returning the type that is the result.
*/
static const Type *lApplyTypeQualifiers(int typeQualifiers, const Type *type, SourcePos pos) {
    if (type == nullptr) {
        return nullptr;
    }

    if ((typeQualifiers & TYPEQUAL_CONST) != 0) {
        type = type->GetAsConstType();
    }

    if (((typeQualifiers & TYPEQUAL_UNIFORM) != 0) && ((typeQualifiers & TYPEQUAL_VARYING) != 0)) {
        Error(pos, "Type \"%s\" cannot be qualified with both uniform and varying.", type->GetString().c_str());
    }

    if ((typeQualifiers & TYPEQUAL_UNIFORM) != 0) {
        if (type->IsVoidType()) {
            Error(pos, "\"uniform\" qualifier is illegal with \"void\" type.");
        } else {
            type = type->GetAsUniformType();
        }
    } else if ((typeQualifiers & TYPEQUAL_VARYING) != 0) {
        if (type->IsVoidType()) {
            Error(pos, "\"varying\" qualifier is illegal with \"void\" type.");
        } else {
            type = type->GetAsVaryingType();
        }
    } else {
        if (type->IsVoidType() == false) {
            type = type->GetAsUnboundVariabilityType();
        }
    }

    if ((typeQualifiers & TYPEQUAL_UNSIGNED) != 0) {
        if ((typeQualifiers & TYPEQUAL_SIGNED) != 0) {
            Error(pos, "Illegal to apply both \"signed\" and \"unsigned\" "
                       "qualifiers.");
        }

        const Type *unsignedType = type->GetAsUnsignedType();
        if (unsignedType != nullptr) {
            type = unsignedType;
        } else {
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
// Attributes

AttrArgument::AttrArgument() : kind(ATTR_ARG_UNKNOWN), intVal(0), stringVal() {}
AttrArgument::AttrArgument(int64_t i) : kind(ATTR_ARG_UINT32), intVal(i), stringVal() {}
AttrArgument::AttrArgument(const std::string &s) : kind(ATTR_ARG_STRING), intVal(0), stringVal(s) {}

void AttrArgument::Print() const {
    switch (kind) {
    case ATTR_ARG_UINT32:
        printf("(%" PRId64 ")", intVal);
        break;
    case ATTR_ARG_STRING:
        printf("(\"%s\")", stringVal.c_str());
        break;
    case ATTR_ARG_UNKNOWN:
        printf("(unknown)");
        break;
    }
}

Attribute::Attribute(const std::string &n) : name(n), arg() {}
Attribute::Attribute(const std::string &n, AttrArgument a) : name(n), arg(a) {}
Attribute::Attribute(const Attribute &a) : name(a.name), arg(a.arg) {}

bool Attribute::IsKnownAttribute() const {
    // Known/supported attributes.
    static std::unordered_set<std::string> lKnownParamAttrs = {"noescape", "address_space", "unmangled", "memory",
                                                               "cdecl",    "external_only", "deprecated"};

    if (lKnownParamAttrs.find(name) != lKnownParamAttrs.end()) {
        return true;
    }

    return false;
}

void Attribute::Print() const {
    printf("%s", name.c_str());
    arg.Print();
}

AttributeList::AttributeList() {}

AttributeList::AttributeList(const AttributeList &attrList) {
    for (const auto &attr : attrList.attributes) {
        AddAttribute(*attr);
    }
}

AttributeList::~AttributeList() {
    for (const auto &attr : attributes) {
        delete attr;
    }
}

void AttributeList::AddAttribute(const Attribute &a) {
    // Create a copy of the given attribute that it owns.
    Attribute *copy = new Attribute(a);
    attributes.push_back(copy);
}

bool AttributeList::HasAttribute(const std::string &name) const {
    for (const auto &attr : attributes) {
        if (attr->name == name) {
            return true;
        }
    }
    return false;
}

Attribute *AttributeList::GetAttribute(const std::string &name) const {
    for (const auto &attr : attributes) {
        if (attr->name == name) {
            return attr;
        }
    }
    return nullptr;
}

void AttributeList::MergeAttrList(const AttributeList &attrList) {
    // TODO: consider issuing a warning if the same attribute is specified
    // several times or with different arguments.
    for (const auto &attr : attrList.attributes) {
        if (!HasAttribute(attr->name)) {
            AddAttribute(*attr);
        }
    }
}

void AttributeList::CheckForUnknownAttributes(SourcePos pos) const {
    for (const auto &attr : attributes) {
        if (!attr->IsKnownAttribute()) {
            Warning(pos, "Ignoring unknown attribute \"%s\".", attr->name.c_str());
        }
    }
}

void AttributeList::Print() const {
    for (const auto &attr : attributes) {
        attr->Print();
        printf(", ");
    }
}

///////////////////////////////////////////////////////////////////////////
// DeclSpecs

DeclSpecs::DeclSpecs(const Type *t, StorageClass sc, int tq) {
    baseType = t;
    storageClass = sc;
    typeQualifiers = tq;
    soaWidth = 0;
    vectorSize = std::monostate{};
    attributeList = nullptr;
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

DeclSpecs::~DeclSpecs() { delete attributeList; }

void DeclSpecs::AddAttrList(const AttributeList &attrList) {
    if (attributeList) {
        attributeList->MergeAttrList(attrList);
        return;
    }
    attributeList = new AttributeList(attrList);
}

const Type *DeclSpecs::GetBaseType(SourcePos pos) const {
    const Type *retType = baseType;

    if (retType == nullptr) {
        Warning(pos, "No type specified in declaration.  Assuming int32.");
        retType = AtomicType::UniformInt32->GetAsUnboundVariabilityType();
    }

    if (std::holds_alternative<int>(vectorSize) || std::holds_alternative<Symbol *>(vectorSize)) {
        const AtomicType *atomicType = CastType<AtomicType>(retType);
        const TemplateTypeParmType *templTypeParam = CastType<TemplateTypeParmType>(retType);
        // Check if the type is valid for vector types
        if (atomicType == nullptr && templTypeParam == nullptr) {
            Error(pos, "Only atomic types (int, float, ...) and template type parameters are legal for vector types.");
            return nullptr;
        }

        if (std::holds_alternative<int>(vectorSize)) {
            // Handle integer vector size
            int size = std::get<int>(vectorSize);
            if (size <= 0) {
                Error(pos, "Illegal to specify vector size of %d.", size);
                return nullptr;
            }
            retType = new VectorType(retType, size);
        } else if (std::holds_alternative<Symbol *>(vectorSize)) {
            // Handle symbol vector size
            Symbol *sym = std::get<Symbol *>(vectorSize);
            if (sym->GetSymbolKind() != Symbol::SymbolKind::TemplateNonTypeParm) {
                Error(pos,
                      "Only atomic types (int, float, ...) and template type parameters are legal for vector types.");
                return nullptr;
            }
            retType = new VectorType(retType, sym);
        } else {
            UNREACHABLE();
        }
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
        } else {
            retType = st->GetAsSOAType(soaWidth);
        }

        if (soaWidth < g->target->getVectorWidth()) {
            PerformanceWarning(pos,
                               "soa<%d> width smaller than gang size %d "
                               "currently leads to inefficient code to access "
                               "soa types.",
                               soaWidth, g->target->getVectorWidth());
        }
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

    if (soaWidth > 0) {
        printf("soa<%d> ", soaWidth);
    }
    lPrintTypeQualifiers(typeQualifiers);
    printf("base type: %s", baseType->GetString().c_str());

    if (attributeList) {
        attributeList->Print();
    }

    if (std::holds_alternative<int>(vectorSize)) {
        printf("<%d>", std::get<int>(vectorSize));
    } else if (std::holds_alternative<Symbol *>(vectorSize)) {
        printf("<%s>", std::get<Symbol *>(vectorSize)->name.c_str());
    } else {
        UNREACHABLE();
    }
    printf("]");
}

///////////////////////////////////////////////////////////////////////////
// Declarator

Declarator::Declarator(DeclaratorKind dk, SourcePos p) : pos(p), kind(dk) {
    child = nullptr;
    typeQualifiers = 0;
    storageClass = SC_NONE;
    arraySize = std::monostate{};
    type = nullptr;
    initExpr = nullptr;
    attributeList = nullptr;
}

Declarator::~Declarator() { delete attributeList; }

void Declarator::InitFromDeclSpecs(DeclSpecs *ds) {
    if (attributeList) {
        attributeList->MergeAttrList(*ds->attributeList);
    } else {
        if (ds->attributeList) {
            attributeList = new AttributeList(*ds->attributeList);
        }
    }

    const Type *baseType = ds->GetBaseType(pos);
    if (!baseType) {
        AssertPos(pos, m->errorCount > 0);
        return;
    }

    if (!lIsFunctionKind(this)) {
        lCheckVariableTypeQualifiers(ds->typeQualifiers, pos);
    }

    InitFromType(baseType, ds);

    if (type == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return;
    }

    if (lIsFunctionKind(this)) {
        if (attributeList) {
            // Check for unknown attributes in function type.
            attributeList->CheckForUnknownAttributes(pos);

            // Handle "address_space" attribute for function return types.
            if (attributeList->HasAttribute("address_space")) {
                auto addrSpace = attributeList->GetAttribute("address_space")->arg.intVal;
                lCheckAddressSpace(addrSpace, name, pos);
                if (auto *ft = CastType<FunctionType>(type)) {
                    auto retType = ft->GetReturnType();
                    auto newRetType = lGetTypeWithAddressSpace(retType, addrSpace, name, pos);
                    type = ft->GetWithReturnType(newRetType);
                }
            }

            // Warn about attributes that are not used for function types.
            if (attributeList->HasAttribute("noescape")) {
                Warning(pos, "Ignoring \"noescape\" attribute for function \"%s\".", name.c_str());
            }
        }
    } else {
        if (attributeList && attributeList->HasAttribute("address_space")) {
            int64_t addrSpace = attributeList->GetAttribute("address_space")->arg.intVal;
            lCheckAddressSpace(addrSpace, name, pos);
            type = lGetTypeWithAddressSpace(type, addrSpace, name, pos);
        }
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
    if (name.size() > 0) {
        printf("%s", name.c_str());
    } else {
        printf("(unnamed)");
    }

    printf(", array size = ");
    if (std::holds_alternative<int>(arraySize)) {
        printf("%d", std::get<int>(arraySize));
    } else if (std::holds_alternative<Symbol *>(arraySize)) {
        printf("%s", std::get<Symbol *>(arraySize)->name.c_str());
    } else {
        UNREACHABLE();
    }

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

    if (attributeList) {
        attributeList->Print();
    }

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
    bool isConst = ((typeQualifiers & TYPEQUAL_CONST) != 0);

    if (hasUniformQual && hasVaryingQual) {
        Error(pos, "Can't provide both \"uniform\" and \"varying\" qualifiers.");
        return;
    }

    lCheckTypeQualifiers(typeQualifiers, kind, pos);

    Variability variability(Variability::Unbound);
    if (hasUniformQual) {
        variability = Variability::Uniform;
    } else if (hasVaryingQual) {
        variability = Variability::Varying;
    }

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
        } else {
            type = ptrType;
        }
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
        } else {
            type = refType;
        }
    } else if (kind == DK_ARRAY) {
        if (baseType->IsVoidType()) {
            Error(pos, "Arrays of \"void\" type are illegal.");
            return;
        }
        if (CastType<ReferenceType>(baseType)) {
            Error(pos, "Arrays of references (type \"%s\") are illegal.", baseType->GetString().c_str());
            return;
        }

        // Check if arraySize holds an int or a Symbol* and create the ArrayType accordingly
        ArrayType *arrayType = nullptr;
        if (std::holds_alternative<int>(arraySize)) {
            int size = std::get<int>(arraySize);
            arrayType = new ArrayType(baseType, size);
        } else if (std::holds_alternative<Symbol *>(arraySize)) {
            Symbol *symbolSize = std::get<Symbol *>(arraySize);
            arrayType = new ArrayType(baseType, symbolSize);
        } else {
            UNREACHABLE();
        }

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
            if (!decl->type->IsTypeDependent()) {
                decl->type = decl->type->ResolveUnboundVariability(Variability::Varying);
            }

            if (d->declSpecs->storageClass != SC_NONE) {
                Error(decl->pos,
                      "Storage class \"%s\" is illegal in "
                      "function parameter declaration for parameter \"%s\".",
                      lGetStorageClassName(d->declSpecs->storageClass), decl->name.c_str());
            }
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
                    if (at->IsUnsized()) {
                        Error(decl->pos, "Arrays with unsized dimensions in "
                                         "dimensions after the first one are illegal in "
                                         "function parameter lists.");
                    }
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
                        if (init == nullptr) {
                            init = llvm::dyn_cast<NullPointerExpr>(decl->initExpr);
                        }
                        if (init == nullptr) {
                            Error(decl->initExpr->pos,
                                  "Default value for parameter "
                                  "\"%s\" must be a compile-time constant.",
                                  decl->name.c_str());
                        }
                    }
                    break;
                } else {
                    decl = decl->child;
                }
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

        if (!returnType->IsTypeDependent()) {
            returnType = returnType->ResolveUnboundVariability(Variability::Varying);
        }

        bool isExternC = ds && (ds->storageClass == SC_EXTERN_C);
        bool isExternSYCL = ds && (ds->storageClass == SC_EXTERN_SYCL);
        bool isExported = ds && ((ds->typeQualifiers & TYPEQUAL_EXPORT) != 0);
        bool isExternalOnly = ds && ds->attributeList && ds->attributeList->HasAttribute("external_only");
        bool isTask = ds && ((ds->typeQualifiers & TYPEQUAL_TASK) != 0);
        bool isUnmasked = ds && ((ds->typeQualifiers & TYPEQUAL_UNMASKED) != 0);
        bool isVectorCall = ds && ((ds->typeQualifiers & TYPEQUAL_VECTORCALL) != 0);
        bool isRegCall = ds && ((ds->typeQualifiers & TYPEQUAL_REGCALL) != 0);
        bool isUnmangled = ds && ds->attributeList && ds->attributeList->HasAttribute("unmangled");
        bool isCdecl = ds && ds->attributeList && ds->attributeList->HasAttribute("cdecl");

        if (!isExported && isExternalOnly) {
            Error(pos, "\"external_only\" attribute is only valid for exported functions.");
            return;
        }

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
        if (isUnmasked && isExported) {
            Warning(pos, "\"unmasked\" qualifier is redundant for exported "
                         "functions.");
        }

        if (isUnmangled) {
            if (isExternC) {
                Error(pos, "Function can't have both \"extern\" \"C\" and \"unmangled\" qualifiers");
                return;
            }
            if (isExternSYCL) {
                Error(pos, "Function can't have both \"extern\" \"SYCL\" and \"unmangled\" qualifiers");
                return;
            }
            if (isExported) {
                Error(pos, "Function can't have both \"export\" and \"unmangled\" qualifiers");
                return;
            }
        }

        if (child == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return;
        }

        const FunctionType *functionType =
            new FunctionType(returnType, args, argNames, argDefaults, argPos, isTask, isExported, isExternalOnly,
                             isExternC, isExternSYCL, isUnmasked, isUnmangled, isVectorCall, isRegCall, isCdecl, pos);

        // handle any explicit __declspecs on the function
        if (ds != nullptr) {
            for (int i = 0; i < (int)ds->declSpecList.size(); ++i) {
                std::string str = ds->declSpecList[i].first;
                SourcePos ds_spec_pos = ds->declSpecList[i].second;

                if (str == "safe") {
                    (const_cast<FunctionType *>(functionType))->isSafe = true;
                } else if (!strncmp(str.c_str(), "cost", 4)) {
                    int cost = atoi(str.c_str() + 4);
                    if (cost < 0) {
                        Error(ds_spec_pos, "Negative function cost %d is illegal.", cost);
                    }
                    (const_cast<FunctionType *>(functionType))->costOverride = cost;
                } else {
                    Error(ds_spec_pos, "__declspec parameter \"%s\" unknown.", str.c_str());
                }
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
    if (dlist != nullptr) {
        declarators = *dlist;
    }
    for (unsigned int i = 0; i < declarators.size(); ++i) {
        if (declarators[i] != nullptr) {
            declarators[i]->InitFromDeclSpecs(declSpecs);
        }
    }
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

        if (decl->type->IsVoidType()) {
            Error(decl->pos, "\"void\" type variable illegal in declaration.");
        } else if (CastType<FunctionType>(decl->type) == nullptr) {
            if (!decl->type->IsTypeDependent()) {
                decl->type = decl->type->ResolveUnboundVariability(Variability::Varying);
            }
            Symbol *sym =
                new Symbol(decl->name, decl->pos, Symbol::SymbolKind::Variable, decl->type, decl->storageClass);

            AttributeList *AL = decl->attributeList;
            if (AL) {
                // Check for unknown attributes for variable declarations.
                AL->CheckForUnknownAttributes(decl->pos);

                if (AL->HasAttribute("noescape")) {
                    Warning(decl->pos, "Ignoring \"noescape\" attribute for variable \"%s\".", decl->name.c_str());
                }
            }

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
        if (ftype == nullptr) {
            continue;
        }

        bool isInline = (declSpecs->typeQualifiers & TYPEQUAL_INLINE);
        bool isNoInline = (declSpecs->typeQualifiers & TYPEQUAL_NOINLINE);
        bool isVectorCall = (declSpecs->typeQualifiers & TYPEQUAL_VECTORCALL);
        bool isRegCall = (declSpecs->typeQualifiers & TYPEQUAL_REGCALL);
        m->AddFunctionDeclaration(decl->name, ftype, decl->storageClass, decl, isInline, isNoInline, isVectorCall,
                                  isRegCall, decl->pos);
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
    std::unordered_set<std::string> seenNames;
    for (unsigned int i = 0; i < sd.size(); ++i) {
        const Type *type = sd[i]->type;
        if (type == nullptr) {
            continue;
        }

        // FIXME: making this fake little DeclSpecs here is really
        // disgusting
        DeclSpecs ds(type);
        if (type->IsVoidType() == false) {
            if (type->IsUniformType()) {
                ds.typeQualifiers |= TYPEQUAL_UNIFORM;
            } else if (type->IsVaryingType()) {
                ds.typeQualifiers |= TYPEQUAL_VARYING;
            } else if (type->GetSOAWidth() != 0) {
                ds.soaWidth = type->GetSOAWidth();
            }
            // FIXME: ds.vectorSize?
        }

        for (unsigned int j = 0; j < sd[i]->declarators->size(); ++j) {
            Declarator *d = (*sd[i]->declarators)[j];
            d->InitFromDeclSpecs(&ds);

            if (d->type->IsVoidType()) {
                Error(d->pos, "\"void\" type illegal for struct member.");
            }

            elementTypes->push_back(d->type);

            if (seenNames.find(d->name) != seenNames.end()) {
                Error(d->pos,
                      "Struct member \"%s\" has same name as a "
                      "previously-declared member.",
                      d->name.c_str());
            } else {
                seenNames.insert(d->name);
            }

            elementNames->push_back(d->name);
            elementPositions->push_back(d->pos);
        }
    }

    for (int i = 0; i < (int)elementTypes->size() - 1; ++i) {
        const ArrayType *arrayType = CastType<ArrayType>((*elementTypes)[i]);

        if (arrayType != nullptr && arrayType->IsUnsized()) {
            Error((*elementPositions)[i], "Unsized arrays aren't allowed except "
                                          "for the last member in a struct definition.");
        }
    }
}
