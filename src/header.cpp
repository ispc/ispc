/*
  Copyright (c) 2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file header.cpp
    @brief This file contains the code to generate the header file, device and
           host stubs, dependency files.
*/

#include "expr.h"
#include "ispc.h"
#include "module.h"
#include "sym.h"
#include "target_enums.h"
#include "type.h"
#include "util.h"

#include <ctype.h>
#include <set>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <utility>
#include <vector>

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Path.h>

using namespace ispc;

/** Given a pointer to an element of a structure, see if it is a struct
    type or an array of a struct type.  If so, return a pointer to the
    underlying struct type. */
static const StructType *lGetElementStructType(const Type *t) {
    const StructType *st = CastType<StructType>(t);
    if (st != nullptr) {
        return st;
    }

    const ArrayType *at = CastType<ArrayType>(t);
    if (at != nullptr) {
        return lGetElementStructType(at->GetElementType());
    }

    return nullptr;
}

static bool lContainsPtrToVarying(const StructType *st) {
    int numElts = st->GetElementCount();

    for (int j = 0; j < numElts; ++j) {
        const Type *t = st->GetElementType(j);

        if (t->IsVaryingType()) {
            return true;
        }
    }

    return false;
}

/** Emits a declaration for the given struct to the given file.  This
    function first makes sure that declarations for any structs that are
    (recursively) members of this struct are emitted first.
 */
static void lEmitStructDecl(const StructType *st, std::vector<const StructType *> *emittedStructs, FILE *file,
                            bool emitUnifs = true) {

    // if we're emitting this for a generic dispatch header file and it's
    // struct that only contains uniforms, don't bother if we're emitting uniforms
    if (!emitUnifs && !lContainsPtrToVarying(st)) {
        return;
    }

    // Has this struct type already been declared?  (This happens if it's a
    // member of another struct for which we emitted a declaration
    // previously.)
    for (int i = 0; i < (int)emittedStructs->size(); ++i) {
        if (Type::EqualIgnoringConst(st, (*emittedStructs)[i])) {
            return;
        }
    }

    // Otherwise first make sure any contained structs have been declared.
    for (int i = 0; i < st->GetElementCount(); ++i) {
        const StructType *elementStructType = lGetElementStructType(st->GetElementType(i));
        if (elementStructType != nullptr) {
            lEmitStructDecl(elementStructType, emittedStructs, file, emitUnifs);
        }
    }

    // And now it's safe to declare this one
    emittedStructs->push_back(st);

    fprintf(file, "#ifndef __ISPC_STRUCT_%s__\n", st->GetCStructName().c_str());
    fprintf(file, "#define __ISPC_STRUCT_%s__\n", st->GetCStructName().c_str());

    char sSOA[48];
    bool pack = false, needsAlign = false;
    llvm::Type *stype = st->LLVMType(g->ctx);
    const llvm::DataLayout *DL = g->target->getDataLayout();
    unsigned int alignment = st->GetAlignment();
    llvm::StructType *stypeStructType = llvm::dyn_cast<llvm::StructType>(stype);

    Assert(stypeStructType);
    if (!(pack = stypeStructType->isPacked())) {
        for (int i = 0; !needsAlign && (i < st->GetElementCount()); ++i) {
            const Type *ftype = st->GetElementType(i)->GetAsNonConstType();
            needsAlign |= ftype->IsVaryingType() && (CastType<StructType>(ftype) == nullptr);
        }
    }
    if (alignment) {
        needsAlign = true;
    }
    if (needsAlign && alignment == 0) {
        alignment = DL->getABITypeAlign(stype).value();
    }
    if (st->GetSOAWidth() > 0) {
        // This has to match the naming scheme in
        // StructType::GetDeclaration().
        snprintf(sSOA, sizeof(sSOA), "_SOA%d", st->GetSOAWidth());
    } else {
        *sSOA = '\0';
    }
    if (!needsAlign) {
        fprintf(file, "%sstruct %s%s {\n", (pack) ? "packed " : "", st->GetCStructName().c_str(), sSOA);
    } else {
        fprintf(file, "__ISPC_ALIGNED_STRUCT__(%u) %s%s {\n", alignment, st->GetCStructName().c_str(), sSOA);
    }
    for (int i = 0; i < st->GetElementCount(); ++i) {
        std::string name = st->GetElementName(i);
        const Type *ftype = st->GetElementType(i)->GetAsNonConstType();
        std::string d_cpp = ftype->GetDeclaration(name, DeclarationSyntax::CPP);
        std::string d_c = ftype->GetDeclaration(name, DeclarationSyntax::C);
        bool same_decls = d_c == d_cpp;

        if (needsAlign && ftype->IsVaryingType() && (CastType<StructType>(ftype) == nullptr)) {
            unsigned uABI = DL->getABITypeAlign(ftype->LLVMStorageType(g->ctx)).value();
            fprintf(file, "    __ISPC_ALIGN__(%u) ", uABI);
        }

        if (!same_decls) {
            fprintf(file, "\n#if defined(__cplusplus)\n");
        }

        // Don't expand arrays, pointers and structures:
        // their insides will be expanded automatically.
        if (!ftype->IsArrayType() && !ftype->IsPointerType() && ftype->IsVaryingType() &&
            (CastType<StructType>(ftype) == nullptr)) {
            fprintf(file, "    %s[%d];\n", d_cpp.c_str(), g->target->getVectorWidth());
            if (!same_decls) {
                fprintf(file,
                        "#else\n"
                        "    %s[%d];\n",
                        d_c.c_str(), g->target->getVectorWidth());
            }
        } else if (CastType<VectorType>(ftype) != nullptr) {
            fprintf(file, "    struct %s;\n", d_cpp.c_str());
            if (!same_decls) {
                fprintf(file,
                        "#else\n"
                        "    struct %s;\n",
                        d_c.c_str());
            }
        } else {
            fprintf(file, "    %s;\n", d_cpp.c_str());
            if (!same_decls) {
                fprintf(file,
                        "#else\n"
                        "    %s;\n",
                        d_c.c_str());
            }
        }

        if (!same_decls) {
            fprintf(file, "#endif // %s field\n", name.c_str());
        }
    }
    fprintf(file, "};\n");
    fprintf(file, "#endif\n\n");
}

/** Given a set of structures that we want to print C declarations of in a
    header file, emit their declarations.
 */
static void lEmitStructDecls(std::vector<const StructType *> &structTypes, FILE *file, bool emitUnifs = true) {
    std::vector<const StructType *> emittedStructs;

    fprintf(file, "\n/* Portable alignment macro that works across different compilers and standards */\n"
                  "#if defined(__cplusplus) && __cplusplus >= 201103L\n"
                  "/* C++11 or newer - use alignas keyword */\n"
                  "#define __ISPC_ALIGN__(x) alignas(x)\n"
                  "#elif defined(__GNUC__) || defined(__clang__)\n"
                  "/* GCC or Clang - use __attribute__ */\n"
                  "#define __ISPC_ALIGN__(x) __attribute__((aligned(x)))\n"
                  "#elif defined(_MSC_VER)\n"
                  "/* Microsoft Visual C++ - use __declspec */\n"
                  "#define __ISPC_ALIGN__(x) __declspec(align(x))\n"
                  "#else\n"
                  "/* Unknown compiler/standard - alignment not supported */\n"
                  "#define __ISPC_ALIGN__(x)\n"
                  "#warning \"Alignment not supported on this compiler\"\n"
                  "#endif // defined(__cplusplus) && __cplusplus >= 201103L\n"
                  "#ifndef __ISPC_ALIGNED_STRUCT__\n"
                  "#if defined(__clang__) || !defined(_MSC_VER)\n"
                  "// Clang, GCC, ICC\n"
                  "#define __ISPC_ALIGNED_STRUCT__(s) struct __ISPC_ALIGN__(s)\n"
                  "#else\n"
                  "// Visual Studio\n"
                  "#define __ISPC_ALIGNED_STRUCT__(s) __ISPC_ALIGN__(s) struct\n"
                  "#endif // defined(__clang__) || !defined(_MSC_VER)\n"
                  "#endif // __ISPC_ALIGNED_STRUCT__\n\n");

    for (unsigned int i = 0; i < structTypes.size(); ++i) {
        lEmitStructDecl(structTypes[i], &emittedStructs, file, emitUnifs);
    }
}

/** Emit C declarations of enumerator types to the generated header file.
 */
static void lEmitEnumDecls(const std::vector<const EnumType *> &enumTypes, FILE *file) {
    if (enumTypes.size() == 0) {
        return;
    }

    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
    fprintf(file, "// Enumerator types with external visibility from ispc code\n");
    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n\n");

    for (unsigned int i = 0; i < enumTypes.size(); ++i) {
        fprintf(file, "#ifndef __ISPC_ENUM_%s__\n", enumTypes[i]->GetEnumName().c_str());
        fprintf(file, "#define __ISPC_ENUM_%s__\n", enumTypes[i]->GetEnumName().c_str());
        std::string declaration = enumTypes[i]->GetDeclaration("", DeclarationSyntax::CPP);
        fprintf(file, "%s {\n", declaration.c_str());

        // Print the individual enumerators
        for (int j = 0; j < enumTypes[i]->GetEnumeratorCount(); ++j) {
            const Symbol *e = enumTypes[i]->GetEnumerator(j);
            Assert(e->constValue != nullptr);
            unsigned int enumValue[1];
            int count = e->constValue->GetValues(enumValue);
            Assert(count == 1);

            // Always print an initializer to set the value.  We could be
            // 'clever' here and detect whether the implicit value given by
            // one plus the previous enumerator value (or zero, for the
            // first enumerator) is the same as the value stored with the
            // enumerator, though that doesn't seem worth the trouble...
            fprintf(file, "    %s = %d%c\n", e->name.c_str(), enumValue[0],
                    (j < enumTypes[i]->GetEnumeratorCount() - 1) ? ',' : ' ');
        }
        fprintf(file, "};\n");
        fprintf(file, "#endif\n\n");
    }
}

/** Print declarations of VectorTypes used in 'export'ed parts of the
    program in the header file.
 */
static void lEmitVectorTypedefs(const std::vector<const VectorType *> &types, FILE *file) {
    if (types.size() == 0) {
        return;
    }

    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
    fprintf(file, "// Vector types with external visibility from ispc code\n");
    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n\n");

    for (unsigned int i = 0; i < types.size(); ++i) {
        std::string baseDecl;
        const SequentialType *vt = types[i]->GetAsNonConstType();
        if (!vt->IsUniformType()) {
            // Varying stuff shouldn't be visibile to / used by the
            // application, so at least make it not simple to access it by
            // not declaring the type here...
            continue;
        }

        int size = vt->GetElementCount();

        llvm::Type *ty = vt->LLVMStorageType(g->ctx);
        int align = g->target->getDataLayout()->getABITypeAlign(ty).value();
        baseDecl = vt->GetBaseType()->GetDeclaration("", DeclarationSyntax::CPP);
        fprintf(file, "#ifndef __ISPC_VECTOR_%s%d__\n", baseDecl.c_str(), size);
        fprintf(file, "#define __ISPC_VECTOR_%s%d__\n", baseDecl.c_str(), size);
        fprintf(file, "#ifdef _MSC_VER\n__declspec( align(%d) ) ", align);
        fprintf(file, "struct %s%d { %s v[%d]; };\n", baseDecl.c_str(), size, baseDecl.c_str(), size);
        fprintf(file, "#else\n");
        fprintf(file, "struct %s%d { %s v[%d]; } __attribute__ ((aligned(%d)));\n", baseDecl.c_str(), size,
                baseDecl.c_str(), size, align);
        fprintf(file, "#endif\n");
        fprintf(file, "#endif\n\n");
    }
    fprintf(file, "\n");
}

/** Add the given type to the vector, if that type isn't already in there.
 */
template <typename T> static void lAddTypeIfNew(const Type *type, std::vector<const T *> *exportedTypes) {
    type = type->GetAsNonConstType();

    // Linear search, so this ends up being n^2.  It's unlikely this will
    // matter in practice, though.
    for (unsigned int i = 0; i < exportedTypes->size(); ++i) {
        if (Type::Equal((*exportedTypes)[i], type)) {
            return;
        }
    }

    const T *castType = CastType<T>(type);
    Assert(castType != nullptr);
    exportedTypes->push_back(castType);
}

/** Given an arbitrary type that appears in the app/ispc interface, add it
    to an appropriate vector if it is a struct, enum, or short vector type.
    Then, if it's a struct, recursively process its members to do the same.
 */
static void lGetExportedTypes(const Type *type, std::vector<const StructType *> *exportedStructTypes,
                              std::vector<const EnumType *> *exportedEnumTypes,
                              std::vector<const VectorType *> *exportedVectorTypes) {
    const ArrayType *arrayType = CastType<ArrayType>(type);
    const StructType *structType = CastType<StructType>(type);
    const FunctionType *ftype = CastType<FunctionType>(type);

    if (CastType<ReferenceType>(type) != nullptr) {
        lGetExportedTypes(type->GetReferenceTarget(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
    } else if (CastType<PointerType>(type) != nullptr) {
        lGetExportedTypes(type->GetBaseType(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
    } else if (arrayType != nullptr) {
        lGetExportedTypes(arrayType->GetElementType(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
    } else if (structType != nullptr) {
        lAddTypeIfNew(type, exportedStructTypes);
        for (int i = 0; i < structType->GetElementCount(); ++i) {
            lGetExportedTypes(structType->GetElementType(i), exportedStructTypes, exportedEnumTypes,
                              exportedVectorTypes);
        }
    } else if (CastType<UndefinedStructType>(type) != nullptr) {
        // do nothing
        ;
    } else if (CastType<EnumType>(type) != nullptr) {
        lAddTypeIfNew(type, exportedEnumTypes);
    } else if (CastType<VectorType>(type) != nullptr) {
        lAddTypeIfNew(type, exportedVectorTypes);
    } else if (ftype != nullptr) {
        // Handle Return Types
        lGetExportedTypes(ftype->GetReturnType(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);

        // And now the parameter types...
        for (int j = 0; j < ftype->GetNumParameters(); ++j) {
            lGetExportedTypes(ftype->GetParameterType(j), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
        }
    } else {
        Assert(CastType<AtomicType>(type) != nullptr);
    }
}

/** Given a set of functions, return the set of structure and vector types
    present in the parameters to them.
 */
static void lGetExportedParamTypes(const std::vector<Symbol *> &funcs,
                                   std::vector<const StructType *> *exportedStructTypes,
                                   std::vector<const EnumType *> *exportedEnumTypes,
                                   std::vector<const VectorType *> *exportedVectorTypes) {
    for (unsigned int i = 0; i < funcs.size(); ++i) {
        const FunctionType *ftype = CastType<FunctionType>(funcs[i]->type);
        Assert(ftype != nullptr);

        // Handle the return type
        lGetExportedTypes(ftype->GetReturnType(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);

        // And now the parameter types...
        for (int j = 0; j < ftype->GetNumParameters(); ++j) {
            lGetExportedTypes(ftype->GetParameterType(j), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
        }
    }
}

static void lPrintFunctionDeclarations(FILE *file, const std::vector<Symbol *> &funcs, bool useExternC = 1,
                                       bool rewriteForDispatch = false) {
    if (useExternC) {
        fprintf(file, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\nextern "
                      "\"C\" {\n#endif // __cplusplus\n");
    }
    // fprintf(file, "#ifdef __cplusplus\nextern \"C\" {\n#endif // __cplusplus\n");
    for (unsigned int i = 0; i < funcs.size(); ++i) {
        const FunctionType *ftype = CastType<FunctionType>(funcs[i]->type);
        Assert(ftype);
        std::string c_decl, cpp_decl;
        std::string fname = funcs[i]->name;
        if (g->calling_conv == CallingConv::x86_vectorcall) {
            fname = "__vectorcall " + fname;
        }
        if (rewriteForDispatch) {
            c_decl = ftype->GetDeclarationForDispatch(fname, DeclarationSyntax::C);
            cpp_decl = ftype->GetDeclarationForDispatch(fname, DeclarationSyntax::CPP);
        } else {
            c_decl = ftype->GetDeclaration(fname, DeclarationSyntax::C);
            cpp_decl = ftype->GetDeclaration(fname, DeclarationSyntax::CPP);
        }
        if (c_decl == cpp_decl) {
            fprintf(file, "    extern %s;\n", c_decl.c_str());
        } else {
            fprintf(file,
                    "#if defined(__cplusplus)\n"
                    "    extern %s;\n",
                    cpp_decl.c_str());
            fprintf(file,
                    "#else\n"
                    "    extern %s;\n",
                    c_decl.c_str());
            fprintf(file, "#endif // %s function declaraion\n", fname.c_str());
        }
    }
    if (useExternC) {
        fprintf(file, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\n} /* end "
                      "extern C */\n#endif // __cplusplus\n");
    }
}

static bool lIsExported(const Symbol *sym) {
    const FunctionType *ft = CastType<FunctionType>(sym->type);
    Assert(ft);
    return ft->IsExported();
}

static bool lIsExternC(const Symbol *sym) {
    const FunctionType *ft = CastType<FunctionType>(sym->type);
    Assert(ft);
    return ft->IsExternC();
}

static void lUnescapeStringInPlace(std::string &str) {
    // There are many more escape sequences, but since this is a path,
    // we can get away with only supporting the basic ones (i.e. no
    // octal, hexadecimal or unicode values).
    for (std::string::iterator it = str.begin(); it != str.end(); ++it) {
        size_t pos = it - str.begin();
        std::string::iterator next = it + 1;
        if (*it == '\\' && next != str.end()) {
            switch (*next) {
#define UNESCAPE_SEQ(c, esc)                                                                                           \
    case c:                                                                                                            \
        *it = esc;                                                                                                     \
        str.erase(next);                                                                                               \
        it = str.begin() + pos;                                                                                        \
        break
                UNESCAPE_SEQ('\'', '\'');
                UNESCAPE_SEQ('?', '?');
                UNESCAPE_SEQ('\\', '\\');
                UNESCAPE_SEQ('a', '\a');
                UNESCAPE_SEQ('b', '\b');
                UNESCAPE_SEQ('f', '\f');
                UNESCAPE_SEQ('n', '\n');
                UNESCAPE_SEQ('r', '\r');
                UNESCAPE_SEQ('t', '\t');
                UNESCAPE_SEQ('v', '\v');
#undef UNESCAPE_SEQ
            }
        }
    }
}

std::string Module::Output::DepsTargetName(const char *srcFile) const {
    if (!depsTarget.empty()) {
        return depsTarget;
    }
    if (!out.empty()) {
        return out;
    }
    if (!IsStdin(srcFile)) {
        std::string targetName = srcFile;
        size_t dot = targetName.find_last_of('.');
        if (dot != std::string::npos) {
            targetName.erase(dot, std::string::npos);
        }
        return targetName + ".o";
    }
    return "a.out";
}

/**
 * This function creates a dependency file listing all headers and source files that
 * the current module depends on. The output can be in one of two formats:
 * 1. A Makefile rule (when makeRuleDeps is true) with the target depending on all files
 * 2. A flat list of all dependencies (when makeRuleDeps is false)
 *
 * The output can be written to a specified file or to stdout.
 *
 * @note In the case of dispatcher module generation, the customOutput
 * parameter should be used instead of the class member output, as the
 * dispatcher requires different output settings.
 */
bool Module::writeDeps(Output &CO) {
    bool generateMakeRule = CO.flags.isMakeRuleDeps();
    std::string targetName = CO.DepsTargetName(srcFile);

    reportInvalidSuffixWarning(CO.deps, OutputType::Deps);

    if (g->debugPrint) { // We may be passed nullptr for stdout output.
        printf("\nWriting dependencies to file %s\n", CO.deps.c_str());
    }
    FILE *file = !CO.deps.empty() ? fopen(CO.deps.c_str(), "w") : stdout;
    if (!file) {
        perror("fopen");
        return false;
    }

    if (generateMakeRule) {
        fprintf(file, "%s:", targetName.c_str());
        // Rules always emit source first.
        if (srcFile && !IsStdin(srcFile)) {
            fprintf(file, " %s", srcFile);
        }
        std::string unescaped;

        for (std::set<std::string>::const_iterator it = registeredDependencies.begin();
             it != registeredDependencies.end(); ++it) {
            unescaped = *it; // As this is preprocessor output, paths come escaped.
            lUnescapeStringInPlace(unescaped);
            if (srcFile && !IsStdin(srcFile) && 0 == strcmp(srcFile, unescaped.c_str())) {
                // If source has been passed, it's already emitted.
                continue;
            }
            fprintf(file, " \\\n");
            fprintf(file, " %s", unescaped.c_str());
        }
        fprintf(file, "\n");
    } else {
        for (std::set<std::string>::const_iterator it = registeredDependencies.begin();
             it != registeredDependencies.end(); ++it) {
            fprintf(file, "%s\n", it->c_str());
        }
    }
    fclose(file);
    return true;
}

std::string emitOffloadParamStruct(const std::string &paramStructName, const Symbol *sym, const FunctionType *fct) {
    std::stringstream out;
    out << "struct " << paramStructName << " {" << std::endl;

    for (int i = 0; i < fct->GetNumParameters(); i++) {
        const Type *orgParamType = fct->GetParameterType(i);
        if (orgParamType->IsPointerType() || orgParamType->IsArrayType()) {
            /* we're passing pointers separately -- no pointers in that struct... */
            continue;
        }

        // const reference parameters can be passed as copies.
        const Type *paramType = nullptr;
        if (orgParamType->IsReferenceType()) {
            if (!orgParamType->IsConstType()) {
                Error(sym->pos, "When emitting offload-stubs, \"export\"ed functions cannot have non-const "
                                "reference-type parameters.\n");
            }
            const ReferenceType *refType = static_cast<const ReferenceType *>(orgParamType);
            paramType = refType->GetReferenceTarget()->GetAsNonConstType();
        } else {
            paramType = orgParamType->GetAsNonConstType();
        }
        std::string paramName = fct->GetParameterName(i);

        std::string tmpArgDecl = paramType->GetDeclaration(paramName, DeclarationSyntax::CPP);
        out << "   " << tmpArgDecl << ";" << std::endl;
    }

    out << "};" << std::endl;
    return out.str();
}

bool Module::writeDevStub() {
    FILE *file = fopen(output.devStub.c_str(), "w");

    reportInvalidSuffixWarning(output.devStub, OutputType::DevStub);

    if (!file) {
        perror("fopen");
        return false;
    }
    fprintf(file, "//\n// %s\n// (device stubs automatically generated by the ispc compiler.)\n",
            output.devStub.c_str());
    fprintf(file, "// DO NOT EDIT THIS FILE.\n//\n\n");
    fprintf(file, "#include \"ispc/dev/offload.h\"\n\n");

    fprintf(file, "#include <stdint.h>\n\n");

    // Collect single linear arrays of the *exported* functions (we'll
    // treat those as "__kernel"s in IVL -- "extern" functions will only
    // be used for dev-dev function calls; only "export" functions will
    // get exported to the host
    std::vector<Symbol *> exportedFuncs;
    m->symbolTable->GetMatchingFunctions(lIsExported, &exportedFuncs);

    // Get all of the struct, vector, and enumerant types used as function
    // parameters.  These vectors may have repeats.
    std::vector<const StructType *> exportedStructTypes;
    std::vector<const EnumType *> exportedEnumTypes;
    std::vector<const VectorType *> exportedVectorTypes;
    lGetExportedParamTypes(exportedFuncs, &exportedStructTypes, &exportedEnumTypes, &exportedVectorTypes);

    // And print them
    lEmitVectorTypedefs(exportedVectorTypes, file);
    lEmitEnumDecls(exportedEnumTypes, file);
    lEmitStructDecls(exportedStructTypes, file);

    fprintf(file, "#ifdef __cplusplus\n");
    fprintf(file, "namespace ispc {\n");
    fprintf(file, "#endif // __cplusplus\n");

    fprintf(file, "\n");
    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
    fprintf(file, "// Functions exported from ispc code\n");
    fprintf(file, "// (so the dev stub knows what to call)\n");
    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
    lPrintFunctionDeclarations(file, exportedFuncs, true);

    fprintf(file, "#ifdef __cplusplus\n");
    fprintf(file, "}/* end namespace */\n");
    fprintf(file, "#endif // __cplusplus\n");

    fprintf(file, "\n");
    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
    fprintf(file, "// actual dev stubs\n");
    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");

    fprintf(file, "// note(iw): due to some linking issues offload stubs *only* work under C++\n");
    fprintf(file, "extern \"C\" {\n\n");
    for (unsigned int i = 0; i < exportedFuncs.size(); ++i) {
        const Symbol *sym = exportedFuncs[i];
        Assert(sym);
        const FunctionType *fct = CastType<FunctionType>(sym->type);
        Assert(fct);

        if (!fct->GetReturnType()->IsVoidType()) {
            // Error(sym->pos,"When emitting offload-stubs, \"export\"ed functions cannot have non-void return
            // types.\n");
            Warning(sym->pos,
                    "When emitting offload-stubs, ignoring \"export\"ed function with non-void return types.\n");
            continue;
        }

        // -------------------------------------------------------
        // first, emit a struct that holds the parameters
        // -------------------------------------------------------
        std::string paramStructName = std::string("__ispc_dev_stub_") + sym->name;
        std::string paramStruct = emitOffloadParamStruct(paramStructName, sym, fct);
        fprintf(file, "%s\n", paramStruct.c_str());
        // -------------------------------------------------------
        // then, emit a fct stub that unpacks the parameters and pointers
        // -------------------------------------------------------
        fprintf(file,
                "void __ispc_dev_stub_%s(\n"
                "            uint32_t         in_BufferCount,\n"
                "            void**           in_ppBufferPointers,\n"
                "            uint64_t*        in_pBufferLengths,\n"
                "            void*            in_pMiscData,\n"
                "            uint16_t         in_MiscDataLength,\n"
                "            void*            in_pReturnValue,\n"
                "            uint16_t         in_ReturnValueLength)\n",
                sym->name.c_str());
        fprintf(file, "{\n");
        fprintf(file, "  struct %s args;\n  memcpy(&args,in_pMiscData,sizeof(args));\n", paramStructName.c_str());
        std::stringstream funcall;

        funcall << "ispc::" << sym->name << "(";
        for (int i = 0; i < fct->GetNumParameters(); i++) {
            // get param type and make it non-const, so we can write while unpacking
            const Type *paramType = nullptr;
            const Type *orgParamType = fct->GetParameterType(i);
            if (orgParamType->IsReferenceType()) {
                if (!orgParamType->IsConstType()) {
                    Error(sym->pos, "When emitting offload-stubs, \"export\"ed functions cannot have non-const "
                                    "reference-type parameters.\n");
                }
                const ReferenceType *refType = static_cast<const ReferenceType *>(orgParamType);
                paramType = refType->GetReferenceTarget()->GetAsNonConstType();
            } else {
                paramType = orgParamType->GetAsNonConstType();
            }

            std::string paramName = fct->GetParameterName(i);

            if (i) {
                funcall << ", ";
            }
            std::string tmpArgName = std::string("_") + paramName;
            if (paramType->IsPointerType() || paramType->IsArrayType()) {
                std::string tmpArgDecl = paramType->GetDeclaration(tmpArgName, DeclarationSyntax::CPP);
                fprintf(file, "  %s;\n", tmpArgDecl.c_str());
                fprintf(file, "  (void *&)%s = ispc_dev_translate_pointer(*in_ppBufferPointers++);\n",
                        tmpArgName.c_str());
                funcall << tmpArgName;
            } else {
                funcall << "args." << paramName;
            }
        }
        funcall << ");";
        fprintf(file, "  %s\n", funcall.str().c_str());
        fprintf(file, "}\n\n");
    }

    // end extern "C"
    fprintf(file, "}/* end extern C */\n");

    fclose(file);
    return true;
}

bool Module::writeHostStub() {
    FILE *file = fopen(output.hostStub.c_str(), "w");

    reportInvalidSuffixWarning(output.hostStub, OutputType::HostStub);

    if (!file) {
        perror("fopen");
        return false;
    }
    fprintf(file, "//\n// %s\n// (device stubs automatically generated by the ispc compiler.)\n",
            output.hostStub.c_str());
    fprintf(file, "// DO NOT EDIT THIS FILE.\n//\n\n");
    fprintf(file, "#include \"ispc/host/offload.h\"\n\n");
    fprintf(
        file,
        "// note(iw): Host stubs do not get extern C linkage -- dev-side already uses that for the same symbols.\n\n");
    // fprintf(file,"#ifdef __cplusplus\nextern \"C\" {\n#endif // __cplusplus\n");

    fprintf(file, "#ifdef __cplusplus\nnamespace ispc {\n#endif // __cplusplus\n\n");

    // Collect single linear arrays of the *exported* functions (we'll
    // treat those as "__kernel"s in IVL -- "extern" functions will only
    // be used for dev-dev function calls; only "export" functions will
    // get exported to the host
    std::vector<Symbol *> exportedFuncs;
    m->symbolTable->GetMatchingFunctions(lIsExported, &exportedFuncs);

    // Get all of the struct, vector, and enumerant types used as function
    // parameters.  These vectors may have repeats.
    std::vector<const StructType *> exportedStructTypes;
    std::vector<const EnumType *> exportedEnumTypes;
    std::vector<const VectorType *> exportedVectorTypes;
    lGetExportedParamTypes(exportedFuncs, &exportedStructTypes, &exportedEnumTypes, &exportedVectorTypes);

    // And print them
    lEmitVectorTypedefs(exportedVectorTypes, file);
    lEmitEnumDecls(exportedEnumTypes, file);
    lEmitStructDecls(exportedStructTypes, file);

    fprintf(file, "\n");
    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
    fprintf(file, "// host-side stubs for dev-side ISPC fucntion(s)\n");
    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
    for (unsigned int i = 0; i < exportedFuncs.size(); ++i) {
        const Symbol *sym = exportedFuncs[i];
        Assert(sym);
        const FunctionType *fct = CastType<FunctionType>(sym->type);
        Assert(fct);

        if (!fct->GetReturnType()->IsVoidType()) {
            Warning(sym->pos,
                    "When emitting offload-stubs, ignoring \"export\"ed function with non-void return types.\n");
            continue;
        }

        // -------------------------------------------------------
        // first, emit a struct that holds the parameters
        // -------------------------------------------------------
        std::string paramStructName = std::string("__ispc_dev_stub_") + sym->name;
        std::string paramStruct = emitOffloadParamStruct(paramStructName, sym, fct);
        fprintf(file, "%s\n", paramStruct.c_str());
        // -------------------------------------------------------
        // then, emit a fct stub that unpacks the parameters and pointers
        // -------------------------------------------------------

        std::string decl = fct->GetDeclaration(sym->name, DeclarationSyntax::CPP);
        fprintf(file, "extern %s {\n", decl.c_str());
        int numPointers = 0;
        fprintf(file, "  %s __args;\n", paramStructName.c_str());

        // ------------------------------------------------------------------
        // write args, and save pointers for later
        // ------------------------------------------------------------------
        std::stringstream pointerArgs;
        for (int i = 0; i < fct->GetNumParameters(); i++) {
            const Type *orgParamType = fct->GetParameterType(i);
            std::string paramName = fct->GetParameterName(i);
            if (orgParamType->IsPointerType() || orgParamType->IsArrayType()) {
                /* we're passing pointers separately -- no pointers in that struct... */
                if (numPointers) {
                    pointerArgs << ",";
                }
                pointerArgs << "(void*)" << paramName;
                numPointers++;
                continue;
            }

            fprintf(file, "  __args.%s = %s;\n", paramName.c_str(), paramName.c_str());
        }
        // ------------------------------------------------------------------
        // writer pointer list
        // ------------------------------------------------------------------
        if (numPointers == 0) {
            pointerArgs << "NULL";
        }
        fprintf(file, "  void *ptr_args[] = { %s };\n", pointerArgs.str().c_str());

        // ------------------------------------------------------------------
        // ... and call the kernel with those args
        // ------------------------------------------------------------------
        fprintf(file, "  static ispc_kernel_handle_t kernel_handle = NULL;\n");
        fprintf(file, "  if (!kernel_handle) kernel_handle = ispc_host_get_kernel_handle(\"__ispc_dev_stub_%s\");\n",
                sym->name.c_str());
        fprintf(file, "  assert(kernel_handle);\n");
        fprintf(file,
                "  ispc_host_call_kernel(kernel_handle,\n"
                "                        &__args, sizeof(__args),\n"
                "                        ptr_args,%i);\n",
                numPointers);
        fprintf(file, "}\n\n");
    }

    // end extern "C"
    fprintf(file, "#ifdef __cplusplus\n");
    fprintf(file, "}/* namespace */\n");
    fprintf(file, "#endif // __cplusplus\n");
    // fprintf(file, "#ifdef __cplusplus\n");
    // fprintf(file, "}/* end extern C */\n");
    // fprintf(file, "#endif // __cplusplus\n");

    fclose(file);
    return true;
}

void Module::writeHeader(FILE *f) {
    // Create a nice guard string from the filename, turning any
    // non-number/letter characters into underbars
    std::string guard = "ISPC_";
    const char *p = output.header.c_str();
    while (*p) {
        if (isdigit(*p)) {
            guard += *p;
        } else if (isalpha(*p)) {
            guard += toupper(*p);
        } else {
            guard += "_";
        }
        ++p;
    }

    if (g->noPragmaOnce) {
        fprintf(f, "#ifndef %s\n#define %s\n\n", guard.c_str(), guard.c_str());
    } else {
        fprintf(f, "#pragma once\n");
    }

    fprintf(f, "#include <stdint.h>\n\n");

    fprintf(f, "#if !defined(__cplusplus)\n"
               "#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)\n"
               "#include <stdbool.h>\n"
               "#else\n"
               "typedef int bool;\n"
               "#endif\n"
               "#endif\n\n");

    if (g->emitInstrumentation) {
        fprintf(f, "#define ISPC_INSTRUMENTATION 1\n");
        fprintf(f, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\nextern \"C\" "
                   "{\n#endif // __cplusplus\n");
        fprintf(f, "  void ISPCInstrument(const char *fn, const char *note, int line, uint64_t mask);\n");
        fprintf(f, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\n} /* end "
                   "extern C */\n#endif // __cplusplus\n");
    }

    // end namespace
    fprintf(f, "\n");
    fprintf(f, "\n#ifdef __cplusplus\nnamespace ispc { /* namespace */\n#endif // __cplusplus\n");

    // Collect single linear arrays of the exported and extern "C"
    // functions
    std::vector<Symbol *> exportedFuncs, externCFuncs;
    m->symbolTable->GetMatchingFunctions(lIsExported, &exportedFuncs);
    m->symbolTable->GetMatchingFunctions(lIsExternC, &externCFuncs);

    // Get all of the struct, vector, and enumerant types used as function
    // parameters.  These vectors may have repeats.
    std::vector<const StructType *> exportedStructTypes;
    std::vector<const EnumType *> exportedEnumTypes;
    std::vector<const VectorType *> exportedVectorTypes;
    lGetExportedParamTypes(exportedFuncs, &exportedStructTypes, &exportedEnumTypes, &exportedVectorTypes);
    lGetExportedParamTypes(externCFuncs, &exportedStructTypes, &exportedEnumTypes, &exportedVectorTypes);

    // Go through the explicitly exported types
    for (int i = 0; i < (int)exportedTypes.size(); ++i) {
        if (const StructType *st = CastType<StructType>(exportedTypes[i].first)) {
            exportedStructTypes.push_back(CastType<StructType>(st->GetAsUniformType()));
        } else if (const EnumType *et = CastType<EnumType>(exportedTypes[i].first)) {
            exportedEnumTypes.push_back(CastType<EnumType>(et->GetAsUniformType()));
        } else if (const VectorType *vt = CastType<VectorType>(exportedTypes[i].first)) {
            exportedVectorTypes.push_back(CastType<VectorType>(vt->GetAsUniformType()));
        } else {
            FATAL("Unexpected type in export list");
        }
    }

    // And print them
    lEmitVectorTypedefs(exportedVectorTypes, f);
    lEmitEnumDecls(exportedEnumTypes, f);
    lEmitStructDecls(exportedStructTypes, f);

    // emit function declarations for exported stuff...
    if (exportedFuncs.size() > 0) {
        fprintf(f, "\n");
        fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
        fprintf(f, "// Functions exported from ispc code\n");
        fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
        lPrintFunctionDeclarations(f, exportedFuncs);
    }

    // end namespace
    fprintf(f, "\n");
    fprintf(f, "\n#ifdef __cplusplus\n} /* namespace */\n#endif // __cplusplus\n");

    // end guard
    if (g->noPragmaOnce) {
        fprintf(f, "\n#endif // %s\n", guard.c_str());
    }
}

bool Module::writeHeader() {
    FILE *f = fopen(output.header.c_str(), "w");

    reportInvalidSuffixWarning(output.header, OutputType::Header);

    if (!f) {
        perror("fopen");
        return false;
    }

    fprintf(f, "//\n// %s\n// (Header automatically generated by the ispc compiler.)\n", output.header.c_str());
    fprintf(f, "// DO NOT EDIT THIS FILE.\n//\n\n");

    writeHeader(f);

    fclose(f);
    return true;
}

bool Module::DispatchHeaderInfo::initialize(std::string headerFileName) {
    EmitUnifs = true;
    EmitFuncs = true;
    EmitFrontMatter = true;
    // This is toggled later.
    EmitBackMatter = false;
    Emit4 = true;
    Emit8 = true;
    Emit16 = true;
    header = std::move(headerFileName);
    fn = header.c_str();

    if (!header.empty()) {
        file = fopen(header.c_str(), "w");
        if (!file) {
            perror("fopen");
            return false;
        }
    }
    return true;
}

void Module::DispatchHeaderInfo::closeFile() {
    if (file != nullptr) {
        fclose(file);
        file = nullptr;
    }
}

bool Module::writeDispatchHeader(DispatchHeaderInfo *DHI) {
    FILE *f = DHI->file;

    reportInvalidSuffixWarning(DHI->fn, OutputType::Header);

    if (DHI->EmitFrontMatter) {
        fprintf(f, "//\n// %s\n// (Header automatically generated by the ispc compiler.)\n", DHI->fn);
        fprintf(f, "// DO NOT EDIT THIS FILE.\n//\n\n");
    }
    // Create a nice guard string from the filename, turning any
    // non-number/letter characters into underbars
    std::string guard = "ISPC_";
    const char *p = DHI->fn;
    while (*p) {
        if (isdigit(*p)) {
            guard += *p;
        } else if (isalpha(*p)) {
            guard += toupper(*p);
        } else {
            guard += "_";
        }
        ++p;
    }
    if (DHI->EmitFrontMatter) {
        if (g->noPragmaOnce) {
            fprintf(f, "#ifndef %s\n#define %s\n\n", guard.c_str(), guard.c_str());
        } else {
            fprintf(f, "#pragma once\n");
        }

        fprintf(f, "#include <stdint.h>\n\n");

        if (g->emitInstrumentation) {
            fprintf(f, "#define ISPC_INSTRUMENTATION 1\n");
            fprintf(f, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\nextern "
                       "\"C\" {\n#endif // __cplusplus\n");
            fprintf(f, "  void ISPCInstrument(const char *fn, const char *note, int line, uint64_t mask);\n");
            fprintf(f, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\n} /* end "
                       "extern C */\n#endif // __cplusplus\n");
        }

        // end namespace
        fprintf(f, "\n");
        fprintf(f, "\n#ifdef __cplusplus\nnamespace ispc { /* namespace */\n#endif // __cplusplus\n\n");
        DHI->EmitFrontMatter = false;
    }

    // Collect single linear arrays of the exported and extern "C"
    // functions
    std::vector<Symbol *> exportedFuncs, externCFuncs;
    m->symbolTable->GetMatchingFunctions(lIsExported, &exportedFuncs);
    m->symbolTable->GetMatchingFunctions(lIsExternC, &externCFuncs);

    int programCount = g->target->getVectorWidth();

    if ((DHI->Emit4 && (programCount == 4)) || (DHI->Emit8 && (programCount == 8)) ||
        (DHI->Emit16 && (programCount == 16))) {
        // Get all of the struct, vector, and enumerant types used as function
        // parameters.  These vectors may have repeats.
        std::vector<const StructType *> exportedStructTypes;
        std::vector<const EnumType *> exportedEnumTypes;
        std::vector<const VectorType *> exportedVectorTypes;
        lGetExportedParamTypes(exportedFuncs, &exportedStructTypes, &exportedEnumTypes, &exportedVectorTypes);
        lGetExportedParamTypes(externCFuncs, &exportedStructTypes, &exportedEnumTypes, &exportedVectorTypes);

        // TODO!: Why there are two almost identical piece of code like this?
        // Go through the explicitly exported types
        for (int i = 0; i < (int)exportedTypes.size(); ++i) {
            if (const StructType *st = CastType<StructType>(exportedTypes[i].first)) {
                exportedStructTypes.push_back(CastType<StructType>(st->GetAsUniformType()));
            } else if (const EnumType *et = CastType<EnumType>(exportedTypes[i].first)) {
                exportedEnumTypes.push_back(CastType<EnumType>(et->GetAsUniformType()));
            } else if (const VectorType *vt = CastType<VectorType>(exportedTypes[i].first)) {
                exportedVectorTypes.push_back(CastType<VectorType>(vt->GetAsUniformType()));
            } else {
                FATAL("Unexpected type in export list");
            }
        }

        // And print them
        if (DHI->EmitUnifs) {
            lEmitVectorTypedefs(exportedVectorTypes, f);
            lEmitEnumDecls(exportedEnumTypes, f);
        }
        lEmitStructDecls(exportedStructTypes, f, DHI->EmitUnifs);

        // Update flags
        DHI->EmitUnifs = false;
        if (programCount == 4) {
            DHI->Emit4 = false;
        } else if (programCount == 8) {
            DHI->Emit8 = false;
        } else if (programCount == 16) {
            DHI->Emit16 = false;
        }
    }
    if (DHI->EmitFuncs) {
        // emit function declarations for exported stuff...
        if (exportedFuncs.size() > 0) {
            fprintf(f, "\n");
            fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
            fprintf(f, "// Functions exported from ispc code\n");
            fprintf(f, "///////////////////////////////////////////////////////////////////////////\n");
            lPrintFunctionDeclarations(f, exportedFuncs, 1, true);
            fprintf(f, "\n");
        }
        DHI->EmitFuncs = false;
    }

    if (DHI->EmitBackMatter) {
        // end namespace
        fprintf(f, "\n");
        fprintf(f, "\n#ifdef __cplusplus\n} /* namespace */\n#endif // __cplusplus\n");

        // end guard
        if (g->noPragmaOnce) {
            fprintf(f, "\n#endif // %s\n", guard.c_str());
        }
        DHI->EmitBackMatter = false;
    }
    return true;
}

std::string lGetNanobindType(const Type *type) {
    if (type->IsAtomicType()) {
        return type->GetDeclaration("", DeclarationSyntax::CPP);
    } else if (type->IsPointerType()) {
        // Both TYPE a[] and TYPE *uniform are represented as pointer types at
        // this point, except structures. We suppose that they are passed from
        // Python side as numpy arrays.
        if (const StructType *ST = CastType<StructType>(type->GetBaseType())) {
            return ST->GetCStructName() + "*";
        } else {
            return "nb::ndarray<" + lGetNanobindType(type->GetBaseType()) + ">";
        }
    } else if (type->IsReferenceType()) {
        return type->GetDeclaration("", DeclarationSyntax::CPP);
    } else if (const StructType *ST = CastType<StructType>(type)) {
        return ST->GetCStructName();
    } else if (const EnumType *ET = CastType<EnumType>(type)) {
        return ET->GetEnumName();
    } else {
        Error(type->GetSourcePos(), "Unsupported type for nanobind wrapper");
        return "";
    }
}

std::string lGetParameters(const FunctionType *ftype) {
    std::string params;
    for (int i = 0; i < ftype->GetNumParameters(); i++) {
        if (i > 0) {
            params += ", ";
        }
        const Type *paramType = ftype->GetParameterType(i);
        const std::string paramName = ftype->GetParameterName(i);
        params += lGetNanobindType(paramType) + " " + paramName;
    }
    return params;
}

std::string lGetCallArguments(const FunctionType *ftype) {
    std::string callArgs;
    for (int i = 0; i < ftype->GetNumParameters(); i++) {
        if (i > 0) {
            callArgs += ", ";
        }
        const std::string paramName = ftype->GetParameterName(i);
        callArgs += paramName;
        const Type *PT = ftype->GetParameterType(i);
        if (PT->IsPointerType() && !CastType<StructType>(PT->GetBaseType())) {
            callArgs += ".data()";
        }
    }
    return callArgs;
}

void lEmitNanobindWrapper(FILE *f, const Symbol *sym) {
    std::string fname = sym->name;
    const FunctionType *ftype = CastType<FunctionType>(sym->type);
    if (!ftype) {
        Error(sym->pos, "Symbol %s is not a function type", fname.c_str());
        return;
    }

    const Type *returnType = ftype->GetReturnType();
    std::string nbWrapper = lGetNanobindType(returnType) + " " + fname + "(" + lGetParameters(ftype) + ")";
    std::string callArgs = lGetCallArguments(ftype);
    fprintf(f, "%s {\n", nbWrapper.c_str());
    if (!returnType->IsVoidType()) {
        fprintf(f, "  return ");
    }
    fprintf(f, "  ispc::%s(%s);\n", fname.c_str(), callArgs.c_str());
    fprintf(f, "}\n");
}

void lEmitNanobindStruct(FILE *f, const StructType *stType) {
    std::string stName = stType->GetCStructName();
    fprintf(f, "  nb::class_<%s>(m, \"%s\")\n", stName.c_str(), stName.c_str());
    fprintf(f, "    .def(nb::init())\n");

    for (int i = 0; i < stType->GetElementCount(); i++) {
        const Type *elemType = stType->GetElementType(i);
        const std::string elemName = stType->GetElementName(i);
        if (elemType->IsArrayType() || elemType->IsVaryingAtomic()) {
            // We need to bind array as numpy arrays
            fprintf(f,
                    "    .def_prop_rw(\"%s\",\n"
                    "      make_array_getter(&%s::%s),\n"
                    "      make_array_setter(&%s::%s))\n",
                    elemName.c_str(), stName.c_str(), elemName.c_str(), stName.c_str(), elemName.c_str());
        } else {
            fprintf(f, "    .def_rw(\"%s\", &%s::%s)\n", elemName.c_str(), stName.c_str(), elemName.c_str());
        }
    }
    fprintf(f, "  ;\n");
}

void lEmitNanobindEnum(FILE *f, const EnumType *etType) {
    std::string etName = etType->GetEnumName();
    fprintf(f, "  nb::enum_<%s>(m, \"%s\")\n", etName.c_str(), etName.c_str());
    for (int i = 0; i < etType->GetEnumeratorCount(); i++) {
        const std::string val = etType->GetEnumerator(i)->name;
        fprintf(f, "    .value(\"%s\", %s::%s)\n", val.c_str(), etName.c_str(), val.c_str());
    }
    fprintf(f, "  ;\n");
}

void lEmitArrayGettersSetters(FILE *f) {
    fprintf(
        f,
        "#include <algorithm>\n"
        "#include <array>\n"
        "#include <numeric>\n"
        "// Type deduction helpers - use recursive template specialization\n"
        "template<typename T>\n"
        "struct array_traits;\n"
        "\n"
        "// Base case for 1D arrays\n"
        "template<typename ElementType, size_t Dim>\n"
        "struct array_traits<ElementType[Dim]> {\n"
        "    using element_type = ElementType;\n"
        "    static constexpr size_t ndim = 1;\n"
        "    static constexpr std::array<size_t, 1> dimensions = {Dim};\n"
        "    static constexpr size_t total_size = Dim;\n"
        "\n"
        "    static auto make_shape_tuple() {\n"
        "        return nb::make_tuple(Dim);\n"
        "    }\n"
        "};\n"
        "\n"
        "// Specialization for 2D arrays\n"
        "template<typename ElementType, size_t Dim1, size_t Dim2>\n"
        "struct array_traits<ElementType[Dim1][Dim2]> {\n"
        "    using element_type = ElementType;\n"
        "    static constexpr size_t ndim = 2;\n"
        "    static constexpr std::array<size_t, 2> dimensions = {Dim1, Dim2};\n"
        "    static constexpr size_t total_size = Dim1 * Dim2;\n"
        "\n"
        "    static auto make_shape_tuple() {\n"
        "        return nb::make_tuple(Dim1, Dim2);\n"
        "    }\n"
        "};\n"
        "\n"
        "// Specialization for 3D arrays (if needed)\n"
        "template<typename ElementType, size_t Dim1, size_t Dim2, size_t Dim3>\n"
        "struct array_traits<ElementType[Dim1][Dim2][Dim3]> {\n"
        "    using element_type = ElementType;\n"
        "    static constexpr size_t ndim = 3;\n"
        "    static constexpr std::array<size_t, 3> dimensions = {Dim1, Dim2, Dim3};\n"
        "    static constexpr size_t total_size = Dim1 * Dim2 * Dim3;\n"
        "\n"
        "    static auto make_shape_tuple() {\n"
        "        return nb::make_tuple(Dim1, Dim2, Dim3);\n"
        "    }\n"
        "};\n"
        "\n"
        "// Helper function to get numpy dtype string from C++ type\n"
        "template<typename T>\n"
        "constexpr const char* get_numpy_dtype() {\n"
        "    if constexpr (std::is_same_v<T, int8_t>) {\n"
        "        return \"int8\";\n"
        "    } else if constexpr (std::is_same_v<T, uint8_t>) {\n"
        "        return \"uint8\";\n"
        "    } else if constexpr (std::is_same_v<T, int16_t>) {\n"
        "        return \"int16\";\n"
        "    } else if constexpr (std::is_same_v<T, uint16_t>) {\n"
        "        return \"uint16\";\n"
        "    } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) {\n"
        "        return \"int32\";\n"
        "    } else if constexpr (std::is_same_v<T, uint32_t>) {\n"
        "        return \"uint32\";\n"
        "    } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, long long>) {\n"
        "        return \"int64\";\n"
        "    } else if constexpr (std::is_same_v<T, uint64_t>) {\n"
        "        return \"uint64\";\n"
        "    } else if constexpr (std::is_same_v<T, float>) {\n"
        "        return \"float32\";\n"
        "    } else if constexpr (std::is_same_v<T, double>) {\n"
        "        return \"float64\";\n"
        "    } else if constexpr (std::is_same_v<T, long double>) {\n"
        "        return \"longdouble\";\n"
        "    } else {\n"
        "        static_assert(sizeof(T) == 0, \"Unsupported type for numpy dtype conversion\");\n"
        "    }\n"
        "}\n"
        "\n"
        "// Simple copy-based approach\n"
        "template<typename ClassType, typename ArrayType>\n"
        "auto make_array_getter(ArrayType ClassType::*member_ptr) {\n"
        "    using traits = array_traits<ArrayType>;\n"
        "    using element_type = typename traits::element_type;\n"
        "\n"
        "    return [member_ptr](const ClassType& self) -> nb::object {\n"
        "        const auto& arr = self.*member_ptr;\n"
        "\n"
        "        // Import numpy\n"
        "        nb::module_ np = nb::module_::import_(\"numpy\");\n"
        "\n"
        "        // Get the appropriate dtype string\n"
        "        const char* dtype_str = get_numpy_dtype<element_type>();\n"
        "\n"
        "        // Create a Python list structure based on dimensions\n"
        "        if constexpr (traits::ndim == 1) {\n"
        "            nb::list list;\n"
        "            for (size_t i = 0; i < traits::dimensions[0]; ++i) {\n"
        "                list.append(arr[i]);\n"
        "            }\n"
        "            return np.attr(\"array\")(list, nb::arg(\"dtype\") = dtype_str);\n"
        "        } else if constexpr (traits::ndim == 2) {\n"
        "            nb::list outer_list;\n"
        "            for (size_t i = 0; i < traits::dimensions[0]; ++i) {\n"
        "                nb::list inner_list;\n"
        "                for (size_t j = 0; j < traits::dimensions[1]; ++j) {\n"
        "                    inner_list.append(arr[i][j]);\n"
        "                }\n"
        "                outer_list.append(inner_list);\n"
        "            }\n"
        "            return np.attr(\"array\")(outer_list, nb::arg(\"dtype\") = dtype_str);\n"
        "        } else if constexpr (traits::ndim == 3) {\n"
        "            nb::list outer_list;\n"
        "            for (size_t i = 0; i < traits::dimensions[0]; ++i) {\n"
        "                nb::list middle_list;\n"
        "                for (size_t j = 0; j < traits::dimensions[1]; ++j) {\n"
        "                    nb::list inner_list;\n"
        "                    for (size_t k = 0; k < traits::dimensions[2]; ++k) {\n"
        "                        inner_list.append(arr[i][j][k]);\n"
        "                    }\n"
        "                    middle_list.append(inner_list);\n"
        "                }\n"
        "                outer_list.append(middle_list);\n"
        "            }\n"
        "            return np.attr(\"array\")(outer_list, nb::arg(\"dtype\") = dtype_str);\n"
        "        } else {\n"
        "            throw std::runtime_error(\"Unsupported array dimension for getter\");\n"
        "        }\n"
        "    };\n"
        "}\n"
        "\n"
        "template<typename ClassType, typename ArrayType>\n"
        "auto make_array_setter(ArrayType ClassType::*member_ptr) {\n"
        "    using traits = array_traits<ArrayType>;\n"
        "    using element_type = typename traits::element_type;\n"
        "\n"
        "    return [member_ptr](ClassType& self, nb::ndarray<nb::numpy, const element_type> arr) {\n"
        "        // Check dimensions\n"
        "        if (arr.ndim() != traits::ndim) {\n"
        "            throw std::runtime_error(\"Array must have \" + std::to_string(traits::ndim) + \" dimensions\");\n"
        "        }\n"
        "\n"
        "        // Check each dimension size\n"
        "        for (size_t i = 0; i < traits::ndim; ++i) {\n"
        "            if (arr.shape(i) != traits::dimensions[i]) {\n"
        "                throw std::runtime_error(\"Dimension \" + std::to_string(i) +\n"
        "                                       \" must have size \" + std::to_string(traits::dimensions[i]) +\n"
        "                                       \", got \" + std::to_string(arr.shape(i)));\n"
        "            }\n"
        "        }\n"
        "\n"
        "        // Copy data - handle different dimensions\n"
        "        auto& target_arr = self.*member_ptr;\n"
        "        if constexpr (traits::ndim == 1) {\n"
        "            std::copy(arr.data(), arr.data() + traits::total_size, &target_arr[0]);\n"
        "        } else if constexpr (traits::ndim == 2) {\n"
        "            std::copy(arr.data(), arr.data() + traits::total_size, &target_arr[0][0]);\n"
        "        } else if constexpr (traits::ndim == 3) {\n"
        "            std::copy(arr.data(), arr.data() + traits::total_size, &target_arr[0][0][0]);\n"
        "        } else {\n"
        "            throw std::runtime_error(\"Unsupported array dimension for setter\");\n"
        "        }\n"
        "    };\n"
        "}\n");
}

bool lStructsWithArrays(const std::vector<const StructType *> &exportedStructTypes) {
    for (const auto &stType : exportedStructTypes) {
        for (int i = 0; i < stType->GetElementCount(); i++) {
            const Type *elemType = stType->GetElementType(i);
            if (elemType->IsArrayType() || elemType->IsVaryingAtomic()) {
                return true;
            }
        }
    }
    return false;
}

bool Module::writeNanobindWrapper() {
    FILE *f = fopen(output.nbWrap.c_str(), "w");
    if (!f) {
        perror("fopen");
        return false;
    }

    reportInvalidSuffixWarning(output.nbWrap, OutputType::NanobindWrapper);

    // remove pathname and extension from filename
    std::string filename = llvm::sys::path::stem(output.nbWrap).str();
    fprintf(f,
            "//\n// %s\n// (Nanobind wrapper automatically generated by the ispc compiler.)\n"
            "// DO NOT EDIT THIS FILE.\n//\n\n",
            filename.c_str());

    // Emit ISPC header to generate self-contained wrapper cpp file that
    // doesn't depend on any other ISPC byproducts.
    fprintf(f, "\n/////////////////////////////////////////////////////////////////////////////\n");
    fprintf(f, "// ISPC header\n");
    fprintf(f, "/////////////////////////////////////////////////////////////////////////////\n\n");

    // We need to emit the header file with pragma once disabled, so that it
    // won't raise compile warnings of #pragma once is being used in main file.
    // Just temporarily disable it, and then restore it.
    int oldPragmaOnce = g->noPragmaOnce;
    g->noPragmaOnce = true;
    writeHeader(f);
    g->noPragmaOnce = oldPragmaOnce;

    fprintf(f, "\n/////////////////////////////////////////////////////////////////////////////\n");
    fprintf(f, "// Nanobind wrapper body\n");
    fprintf(f, "/////////////////////////////////////////////////////////////////////////////\n\n");
    fprintf(f, "\n#include <nanobind/nanobind.h>\n"
               "#include <nanobind/ndarray.h>\n\n");

    fprintf(f, "namespace nb = nanobind;\n\n");

    std::vector<Symbol *> exportedFuncs;
    m->symbolTable->GetMatchingFunctions(lIsExported, &exportedFuncs);

    std::vector<const StructType *> exportedStructTypes;
    std::vector<const EnumType *> exportedEnumTypes;
    std::vector<const VectorType *> exportedVectorTypes;
    lGetExportedParamTypes(exportedFuncs, &exportedStructTypes, &exportedEnumTypes, &exportedVectorTypes);

    if (lStructsWithArrays(exportedStructTypes)) {
        lEmitArrayGettersSetters(f);
    }

    // We do this to avoid emitting ispc:: every time we need to use struct/enum types.
    fprintf(f, "\n////////////////////////////////////////////////////////////////////////////\n");
    fprintf(f, "// Bring exported ISPC structs and enums into the current scope\n");
    fprintf(f, "////////////////////////////////////////////////////////////////////////////\n\n");
    for (const auto &stType : exportedStructTypes) {
        fprintf(f, "using ispc::%s;\n", stType->GetCStructName().c_str());
    }
    for (const auto &etType : exportedEnumTypes) {
        fprintf(f, "using ispc::%s;\n", etType->GetEnumName().c_str());
    }
    for (const auto &vType : exportedVectorTypes) {
        Error(vType->GetSourcePos(), "Nanobind wrapper generation does not support vector types.\n");
    }

    if (exportedFuncs.size() > 0) {
        fprintf(f, "\n///////////////////////////////////////////////////////////////////////////\n");
        fprintf(f, "// Wrappers for functions exported from ispc code\n");
        fprintf(f, "///////////////////////////////////////////////////////////////////////////\n\n");
    }
    for (const auto &func : exportedFuncs) {
        lEmitNanobindWrapper(f, func);
        fprintf(f, "\n");
    }

    fprintf(f, "\n///////////////////////////////////////////////////////////////////////////\n");
    fprintf(f, "// Python module definition\n");
    fprintf(f, "///////////////////////////////////////////////////////////////////////////\n\n");
    fprintf(f, "NB_MODULE(%s, m) {\n", filename.c_str());
    for (const auto &stType : exportedStructTypes) {
        lEmitNanobindStruct(f, stType);
        fprintf(f, "\n");
    }
    for (const auto &etType : exportedEnumTypes) {
        lEmitNanobindEnum(f, etType);
        fprintf(f, "\n");
    }

    fprintf(f, "\n");
    fprintf(f, "  // Register wrappers\n");
    for (const auto &func : exportedFuncs) {
        std::string fname = func->name;
        fprintf(f, "  m.def(\"%s\", &%s);\n", fname.c_str(), fname.c_str());
    }
    fprintf(f, "}\n");

    fclose(f);

    return true;
}
