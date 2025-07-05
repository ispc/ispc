/*
  Copyright (c) 2010-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file module.cpp
    @brief Impementation of the Module class, which collects the result of compiling
           a source file and then generates output (object files, etc.)
*/

#include "module.h"
#include "builtins.h"
#include "ctx.h"
#include "expr.h"
#include "func.h"
#include "ispc_version.h"
#include "llvmutil.h"
#include "opt.h"
#include "stmt.h"
#include "sym.h"
#include "type.h"
#include "util.h"

#include <algorithm>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <set>
#include <stdarg.h>
#include <stdio.h>
#include <utility>

#include <clang/Basic/Version.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#ifdef ISPC_XE_ENABLED
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#endif

#ifdef ISPC_XE_ENABLED
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#if defined(_WIN64)
#define OCLOC_LIBRARY_NAME "ocloc64.dll"
#elif defined(_WIN32)
#define OCLOC_LIBRARY_NAME "ocloc32.dll"
#elif defined(__linux__)
#define OCLOC_LIBRARY_NAME "libocloc.so"
#else
#error "Unexpected platform"
#endif
#endif

#ifdef ISPC_HOST_IS_WINDOWS
#include <io.h>
// windows.h defines CALLBACK as __stdcall
// clang/Analysis/CFG.h contains typename with name CALLBACK, which is got screwed up.
// So we include it after clang includes.
#include <windows.h>
#endif

using namespace ispc;

// The magic constants are derived from https://github.com/intel/compute-runtime repo
// compute-runtime/shared/source/compiler_interface/intermediate_representations.h
const char *llvmBcMagic = "BC\xc0\xde";
const char *llvmBcMagicWrapper = "\xDE\xC0\x17\x0B";
const char *spirvMagic = "\x07\x23\x02\x03";
const char *spirvMagicInv = "\x03\x02\x23\x07";

static bool lHasSameMagic(const char *expectedMagic, std::ifstream &is) {
    // Check for magic number at the beginning of the file
    std::vector<unsigned char> code;
    is.seekg(0, std::ios::end);
    size_t codeSize = is.tellg();
    is.seekg(0, std::ios::beg);

    auto binaryMagicLen = std::min(strlen(expectedMagic), codeSize);
    code.resize(binaryMagicLen);
    is.read((char *)code.data(), binaryMagicLen);
    is.seekg(0, std::ios::beg);

    std::string magicHeader(code.begin(), code.end());
    return magicHeader.compare(expectedMagic) == 0;
}

static bool lIsLlvmBitcode(std::ifstream &is) {
    return lHasSameMagic(llvmBcMagic, is) || lHasSameMagic(llvmBcMagicWrapper, is);
}

static bool lIsSpirVBitcode(std::ifstream &is) {
    return lHasSameMagic(spirvMagic, is) || lHasSameMagic(spirvMagicInv, is);
}

static bool lIsStdlibPseudoFile(const std::string &name) {
    // This needs to correspond to lAddImplicitInclude
#ifdef ISPC_HOST_IS_WINDOWS
    if (name == "c:/core.isph" || name == "c:/stdlib.isph") {
        return true;
    }
#else
    if (name == "/core.isph" || name == "/stdlib.isph") {
        return true;
    }
#endif // ISPC_HOST_IS_WINDOWS
    return false;
}

/*! this is where the parser tells us that it has seen the given file
    name in the CPP hash */
const char *Module::RegisterDependency(const std::string &fileName) {
    if (fileName[0] != '<' && fileName != "stdlib.ispc" && !lIsStdlibPseudoFile(fileName)) {
        auto res = registeredDependencies.insert(fileName);
        return res.first->c_str();
    } else {
        auto res = pseudoDependencies.insert(fileName);
        return res.first->c_str();
    }
}

static void lDeclareSizeAndPtrIntTypes(SymbolTable *symbolTable) {
    const Type *ptrIntType = (g->target->is32Bit()) ? AtomicType::VaryingInt32 : AtomicType::VaryingInt64;
    ptrIntType = ptrIntType->GetAsUnboundVariabilityType();

    symbolTable->AddType("intptr_t", ptrIntType, SourcePos());
    symbolTable->AddType("uintptr_t", ptrIntType->GetAsUnsignedType(), SourcePos());
    symbolTable->AddType("ptrdiff_t", ptrIntType, SourcePos());

    const Type *sizeType =
        (g->target->is32Bit() || g->opt.force32BitAddressing) ? AtomicType::VaryingUInt32 : AtomicType::VaryingUInt64;
    sizeType = sizeType->GetAsUnboundVariabilityType();
    symbolTable->AddType("size_t", sizeType, SourcePos());
}

/** After compilation completes, there's often a lot of extra debugging
    metadata left around that isn't needed any more--for example, for
    static functions that weren't actually used, function information for
    functions that were inlined, etc.  This function takes a llvm::Module
    and tries to strip out all of this extra stuff.
 */
static void lStripUnusedDebugInfo(llvm::Module *module) { return; }

/** Code model needs to be set twice - it's passed directly to the TargetMachine and it's set for the llvm::Module.
    TargetMachine setting is directly governing the code generation, while setting it for the Module yields in
    metadata node, which is used to ensure that the model setting survives through LTO. We don't necesseraly need
    the latter, but it's possible to use LLVM IR from ISPC for LTO mode or pass it manually to llc.
 */
static void lSetCodeModel(llvm::Module *module) {
    MCModel model = g->target->getMCModel();
    switch (model) {
    case ispc::MCModel::Small:
        module->setCodeModel(llvm::CodeModel::Small);
        break;
    case ispc::MCModel::Large:
        module->setCodeModel(llvm::CodeModel::Large);
        break;
    case ispc::MCModel::Default:
        // Do nothing.
        break;
    }
}

/** We need to set the correct value inside "PIC Level" metadata as it can be
    used by some targets to generate code in the different way. Unlike code
    model, it is not passed to the TargetMachine, so it is the only place for
    codegen to access the correct value. */
static void lSetPICLevel(llvm::Module *module) {
    PICLevel picLevel = g->target->getPICLevel();
    switch (picLevel) {
    case ispc::PICLevel::Default:
        // We're leaving the default. There's a similar case in the LLVM
        // frontend code where they don't set the level when pic flag is
        // omitted in the command line.
        break;
    case ispc::PICLevel::NotPIC:
        module->setPICLevel(llvm::PICLevel::NotPIC);
        break;
    case ispc::PICLevel::SmallPIC:
        module->setPICLevel(llvm::PICLevel::SmallPIC);
        break;
    case ispc::PICLevel::BigPIC:
        module->setPICLevel(llvm::PICLevel::BigPIC);
        break;
    }
}

///////////////////////////////////////////////////////////////////////////
// Module

Module::Module(const char *fn) : srcFile(fn) {
    // It's a hack to do this here, but it must be done after the target
    // information has been set (so e.g. the vector width is known...)  In
    // particular, if we're compiling to multiple targets with different
    // vector widths, this needs to be redone each time through.
    InitLLVMUtil(g->ctx, *g->target);

    symbolTable = new SymbolTable;
    ast = new AST;

    lDeclareSizeAndPtrIntTypes(symbolTable);

    module = new llvm::Module(!IsStdin(srcFile) ? srcFile : "<stdin>", *g->ctx);

#if ISPC_LLVM_VERSION >= ISPC_LLVM_21_0
    module->setTargetTriple(g->target->GetTriple());
#else
    module->setTargetTriple(g->target->GetTriple().str());
#endif
    // DataLayout information supposed to be managed in single place in Target class.
    module->setDataLayout(g->target->getDataLayout()->getStringRepresentation());
    lSetCodeModel(module);
    lSetPICLevel(module);

#if ISPC_LLVM_VERSION >= ISPC_LLVM_19_0 && ISPC_LLVM_VERSION < ISPC_LLVM_21_0
    // LLVM is transitioning to new debug info representation, use "old" style for now.
    module->setIsNewDbgInfoFormat(false);
#endif

    // Version strings.
    // Have ISPC details and LLVM details as two separate strings attached to !llvm.ident.
    llvm::NamedMDNode *identMetadata = module->getOrInsertNamedMetadata("llvm.ident");
    std::string ispcVersion = std::string(ISPC_VERSION_STRING);
    std::string llvmVersion = clang::getClangToolFullVersion("LLVM");

    llvm::Metadata *identNode[] = {llvm::MDString::get(*g->ctx, ispcVersion)};
    identMetadata->addOperand(llvm::MDNode::get(*g->ctx, identNode));
    llvm::Metadata *identNode2[] = {llvm::MDString::get(*g->ctx, llvmVersion)};
    identMetadata->addOperand(llvm::MDNode::get(*g->ctx, identNode2));

    if (g->generateDebuggingSymbols) {
        llvm::TimeTraceScope TimeScope("Create Debug Data");
        // To enable debug information on Windows, we have to let llvm know, that
        // debug information should be emitted in CodeView format.

        switch (g->debugInfoType) {
        case Globals::DebugInfoType::CodeView:
            module->addModuleFlag(llvm::Module::Warning, "CodeView", 1);
            break;
        case Globals::DebugInfoType::DWARF:
            module->addModuleFlag(llvm::Module::Warning, "Dwarf Version", g->generateDWARFVersion);
            break;
        default:
            FATAL("Incorrect debugInfoType");
            break;
        }
        diBuilder = new llvm::DIBuilder(*module);

        // Let the DIBuilder know that we're starting a new compilation
        // unit.
        if (IsStdin(srcFile)) {
            // Unfortunately we can't yet call Error() since the global 'm'
            // variable hasn't been initialized yet.
            Error(SourcePos(), "Can't emit debugging information with no source file on disk.\n");
            ++errorCount;
            delete diBuilder;
            diBuilder = nullptr;
        } else {
            auto [directory, name] = GetDirectoryAndFileName(g->currentDirectory, srcFile);
            auto srcFile = diBuilder->createFile(name, directory);
            // Use DW_LANG_C_plus_plus to avoid problems with debigging on Xe.
            // The debugger reads symbols partially when a solib file is loaded.
            // The kernel name is one of these read symbols. ISPC produces namespace
            // for example "ispc::simple_ispc". Matching the breakpoint location
            // "simple_ispc" with the symbol name fails if module language is C and not C++.
            diCompileUnit =
                diBuilder->createCompileUnit(llvm::dwarf::DW_LANG_C_plus_plus,          /* lang */
                                             srcFile,                                   /* filename */
                                             ispcVersion.c_str(),                       /* producer */
                                             g->opt.level > 0 /* is optimized */, "-g", /* command line args */
                                             0 /* run time version */);
        }
    }
}

Module::Module(const char *filename, Output &output, Module::CompilationMode mode) : Module(filename) {
    this->output = output;
    this->m_compilationMode = mode;
}

std::unique_ptr<Module> Module::Create(const char *srcFile, Output &output, Module::CompilationMode mode) {
    auto ptr = std::make_unique<Module>(srcFile, output, mode);

    // Here, we do not transfer the ownership of the target to the global
    // variable. We just set the observer pointer here.
    m = ptr.get();
    return ptr;
}

Module::~Module() {
    if (symbolTable) {
        delete symbolTable;
    }
    if (ast) {
        delete ast;
    }
    if (module) {
        delete module;
    }
    if (diBuilder) {
        delete diBuilder;
    }
}

extern FILE *yyin;
extern int yyparse();
typedef struct yy_buffer_state *YY_BUFFER_STATE;
extern void yy_switch_to_buffer(YY_BUFFER_STATE);
extern YY_BUFFER_STATE yy_create_buffer(FILE *, int);
extern void yy_delete_buffer(YY_BUFFER_STATE);
extern void ParserInit();

int Module::parse() {
    // No preprocessor, just open up the file if it's not stdin..
    FILE *f = nullptr;
    if (IsStdin(srcFile)) {
        f = stdin;
    } else {
        f = fopen(srcFile, "r");
        if (f == nullptr) {
            perror(srcFile);
            return 1;
        }
    }
    yyin = f;
    yy_switch_to_buffer(yy_create_buffer(yyin, 4096));
    yyparse();
    fclose(f);

    return 0;
}

int Module::CompileFile() {
    llvm::TimeTraceScope CompileFileTimeScope(
        "CompileFile", llvm::StringRef(srcFile + ("_" + std::string(g->target->GetISAString()))));
    ParserInit();

    int pre_stage = PRE_OPT_NUMBER;
    debugDumpModule(module, "Empty", pre_stage++);

    if (g->runCPP) {
        llvm::TimeTraceScope TimeScope("Frontend parser");
        int err = preprocessAndParse();
        if (g->onlyCPP || err) {
            return err;
        }
    } else {
        llvm::TimeTraceScope TimeScope("Frontend parser");
        if (int err = parse()) {
            return err;
        }
    }

    debugDumpModule(module, "Parsed", pre_stage++);

    ast->Print(g->astDump);

    if (g->NoOmitFramePointer) {
        for (llvm::Function &f : *module) {
            f.addFnAttr("frame-pointer", "all");
        }
    }
    for (llvm::Function &f : *module) {
        g->target->markFuncWithTargetAttr(&f);
    }
    ast->GenerateIR();

    debugDumpModule(module, "GenerateIR", pre_stage++);

    if (!g->genStdlib) {
        llvm::TimeTraceScope TimeScope("DefineStdlib");
        LinkStandardLibraries(module, pre_stage);
    }

    for (llvm::Function &f : *module) {
        g->target->markFuncWithTargetAttr(&f);
    }

    if (diBuilder) {
        diBuilder->finalize();
    }

    // Skip optimization for stdlib. We need to consider shipping optimized
    // stdlibs library but at the moment it is not so.
    if (!g->genStdlib) {
        llvm::TimeTraceScope TimeScope("Optimize");
        if (errorCount == 0) {
            Optimize(module, g->opt.level);
        }
    }

    return errorCount;
}

void Module::AddTypeDef(const std::string &name, const Type *type, SourcePos pos) {
    // Typedefs are easy; just add the mapping between the given name and
    // the given type.
    symbolTable->AddType(name.c_str(), type, pos);
}

// Construct ConstExpr as initializer for varying const value from given expression list if possible
// T is bool* or std::vector<llvm::APFloat> or int8_t* (and others integer types) here.
// Although, it looks like very unlogical this is the probably most reasonable
// approach to unify usage of vals variable inside function.
// Approach with T is bool, llvm::APFloat, int8_t, ... doesn't work because
// 1) We can't dereference &vals[i++] when it is std::vector<bool>. The
// specialization of vector with bool doesn't necessarily store its elements as
// contiguous array.
// 2) MSVC doesn't support VLAs, so T vals[N] is not an option.
// 3) T vals[64] is not an option because llvm::APFloat doesn't have the
// default constructor. Same applicable for std::array.
template <class T>
Expr *lCreateConstExpr(ExprList *exprList, const AtomicType::BasicType basicType, const Type *type,
                       const std::string &name, SourcePos pos) {
    const int N = g->target->getVectorWidth();
    bool canConstructConstExpr = true;
    // Exit early if the number of initializers is more than N
    if (exprList->exprs.size() > N) {
        return nullptr;
    }
    using ManagedType =
        typename std::conditional<std::is_pointer<T>::value, std::unique_ptr<typename std::remove_pointer<T>::type[]>,
                                  int // unused placeholder
                                  >::type;
    ManagedType managedMemory;
    T vals;
    if constexpr (std::is_same_v<T, std::vector<llvm::APFloat>>) {
        switch (basicType) {
        case AtomicType::TYPE_FLOAT16:
            vals.resize(N, llvm::APFloat::getZero(llvm::APFloat::IEEEhalf()));
            break;
        case AtomicType::TYPE_FLOAT:
            vals.resize(N, llvm::APFloat::getZero(llvm::APFloat::IEEEsingle()));
            break;
        case AtomicType::TYPE_DOUBLE:
            vals.resize(N, llvm::APFloat::getZero(llvm::APFloat::IEEEdouble()));
            break;
        default:
            return nullptr;
        }
    } else {
        // T equals int8_t* and etc.
        // We allocate PointToType[N] on the heap. It is managed by unique_ptr.
        using PointToType = typename std::remove_pointer<T>::type;
        managedMemory = std::make_unique<PointToType[]>(N);
        vals = managedMemory.get();
        memset(vals, 0, N * sizeof(PointToType));
    }

    int i = 0;
    for (Expr *expr : exprList->exprs) {
        const ConstExpr *ce = llvm::dyn_cast<ConstExpr>(TypeCheckAndOptimize(expr));
        // ConstExpr of length 1 implies that it contains uniform value
        if (!ce || ce->Count() != 1) {
            canConstructConstExpr = false;
            break;
        }

        if constexpr (std::is_same_v<T, std::vector<llvm::APFloat>>) {
            std::vector<llvm::APFloat> ce_vals;
            llvm::Type *to_type = type->GetAsUniformType()->LLVMType(g->ctx);
            ce->GetValues(ce_vals, to_type);
            vals[i++] = ce_vals[0];
        } else {
            ce->GetValues(&vals[i++]);
        }
    }

    if (i == 1) {
        // In case when, e.g., { 1, }, we need to set all values as equaled to
        // the first one to match the behaviour with other initializations.
        while (i < N) {
            vals[i++] = vals[0];
        }
    } else if (i != N) {
        // In other cases when number of values in initializer don't match the
        // vector width, we report error to match the other behaviour.
        Error(pos, "Initializer list for %s \"%s\" must have %d elements (has %d).", name.c_str(),
              type->GetString().c_str(), N, (int)exprList->exprs.size());
    }

    if (canConstructConstExpr) {
        return new ConstExpr(type, vals, pos);
    }
    return nullptr;
}

Expr *lConvertExprListToConstExpr(Expr *initExpr, const Type *type, const std::string &name, SourcePos pos) {
    ExprList *exprList = llvm::dyn_cast<ExprList>(initExpr);
    if (type->IsConstType() && type->IsVaryingType() && type->IsAtomicType()) {
        const AtomicType::BasicType basicType = CastType<AtomicType>(type)->basicType;
        switch (basicType) {
        case AtomicType::TYPE_BOOL:
            return lCreateConstExpr<bool *>(exprList, basicType, type, name, pos);
        case AtomicType::TYPE_INT8:
            return lCreateConstExpr<int8_t *>(exprList, basicType, type, name, pos);
        case AtomicType::TYPE_UINT8:
            return lCreateConstExpr<uint8_t *>(exprList, basicType, type, name, pos);
        case AtomicType::TYPE_INT16:
            return lCreateConstExpr<int16_t *>(exprList, basicType, type, name, pos);
        case AtomicType::TYPE_UINT16:
            return lCreateConstExpr<uint16_t *>(exprList, basicType, type, name, pos);
        case AtomicType::TYPE_INT32:
            return lCreateConstExpr<int32_t *>(exprList, basicType, type, name, pos);
        case AtomicType::TYPE_UINT32:
            return lCreateConstExpr<uint32_t *>(exprList, basicType, type, name, pos);
        case AtomicType::TYPE_INT64:
            return lCreateConstExpr<int64_t *>(exprList, basicType, type, name, pos);
        case AtomicType::TYPE_UINT64:
            return lCreateConstExpr<uint64_t *>(exprList, basicType, type, name, pos);
        case AtomicType::TYPE_FLOAT16:
        case AtomicType::TYPE_FLOAT:
        case AtomicType::TYPE_DOUBLE:
            return lCreateConstExpr<std::vector<llvm::APFloat>>(exprList, basicType, type, name, pos);
        default:
            // Unsupported types.
            break;
        }
    }
    return nullptr;
}

void Module::AddGlobalVariable(Declarator *decl, bool isConst) {
    const std::string &name = decl->name;
    const Type *type = decl->type;
    Expr *initExpr = decl->initExpr;
    StorageClass storageClass = decl->storageClass;
    SourcePos pos = decl->pos;

    // TODO!: what default 0 means here?
    unsigned int alignment = 0;

    // These may be nullptr due to errors in parsing; just gracefully return
    // here if so.
    if (name == "" || type == nullptr) {
        Assert(errorCount > 0);
        return;
    }

    // Check attrbutes for global variables.
    AttributeList *attrList = decl->attributeList;
    if (attrList) {
        // Check for unknown attributes for global variable declarations.
        attrList->CheckForUnknownAttributes(pos);

        if (attrList->HasAttribute("noescape")) {
            Warning(pos, "Ignoring \"noescape\" attribute for global variable \"%s\".", name.c_str());
        }

        // This attribute is provided for the particular variable declaration.
        alignment = attrList->GetAlignedAttrValue(pos);
    }

    // If no alignment was specified for the particular variable declaration,
    // then use the alignment specified for the type.
    if (!alignment) {
        alignment = type->GetAlignment();
    }

    if (symbolTable->LookupFunction(name.c_str())) {
        Error(pos, "Global variable \"%s\" shadows previously-declared function.", name.c_str());
        return;
    }

    if (symbolTable->LookupFunctionTemplate(name.c_str())) {
        Error(pos, "Global variable \"%s\" shadows previously-declared function template.", name.c_str());
        return;
    }

    if (storageClass.IsExternC()) {
        Error(pos, "extern \"C\" qualifier can only be used for functions.");
        return;
    }

    if (storageClass.IsExternSYCL()) {
        Error(pos, "extern \"SYCL\" qualifier can only be used for functions.");
        return;
    }

    if (type->IsVoidType()) {
        Error(pos, "\"void\" type global variable is illegal.");
        return;
    }

    if (!type->IsCompleteType()) {
        Error(pos, "variable \"%s\" has incomplete type \"%s\"", name.c_str(), type->GetString().c_str());
        return;
    }

    type = ArrayType::SizeUnsizedArrays(type, initExpr);
    if (type == nullptr) {
        return;
    }

    const ArrayType *at = CastType<ArrayType>(type);
    if (at != nullptr && at->TotalElementCount() == 0) {
        Error(pos, "Illegal to declare a global variable with unsized "
                   "array dimensions that aren't set with an initializer "
                   "expression.");
        return;
    }

    llvm::Type *llvmType = type->LLVMStorageType(g->ctx);
    if (llvmType == nullptr) {
        return;
    }

    // See if we have an initializer expression for the global.  If so,
    // make sure it's a compile-time constant!
    llvm::Constant *llvmInitializer = nullptr;
    ConstExpr *constValue = nullptr;
    if (storageClass.IsExtern()) {
        if (initExpr != nullptr) {
            Error(pos, "Initializer can't be provided with \"extern\" global variable \"%s\".", name.c_str());
        }
    } else {
        if (initExpr != nullptr) {
            initExpr = TypeCheck(initExpr);
            if (initExpr != nullptr) {
                // We need to make sure the initializer expression is
                // the same type as the global.
                if (llvm::dyn_cast<ExprList>(initExpr) == nullptr) {
                    initExpr = TypeConvertExpr(initExpr, type, "initializer");
                } else {
                    // The alternative is to create ConstExpr initializing
                    // expression with correct type and value from ExprList.
                    // If we have exprList that initalizes const varying int, e.g.:
                    // static const int x = { 0, 1, 2, 3 };
                    // then we need to convert rvalue to varying ConstExpr value.
                    // It will be utilized later in arithmetic expressions that can be
                    // calculated in compile time (Optimize function call below),
                    Expr *ce = lConvertExprListToConstExpr(initExpr, type, name, pos);
                    if (ce) {
                        initExpr = ce;
                    }
                }

                if (initExpr != nullptr) {
                    initExpr = TypeCheckAndOptimize(initExpr);
                    if (initExpr == nullptr) {
                        Error(pos, "Initialization of global variable \"%s\" was unsuccessful.", name.c_str());
                        return;
                    }
                    // Fingers crossed, now let's see if we've got a
                    // constant value..
                    std::pair<llvm::Constant *, bool> initPair = initExpr->GetStorageConstant(type);
                    llvmInitializer = initPair.first;

                    // If compiling for multitarget, skip initialization for
                    // indentified scenarios unless it's static
                    if (llvmInitializer != nullptr) {
                        if (!storageClass.IsStatic() && initPair.second) {
                            if (g->isMultiTargetCompilation == true) {
                                Error(initExpr->pos,
                                      "Initializer for global variable \"%s\" is not a constant for multi-target "
                                      "compilation.",
                                      name.c_str());
                                return;
                            }
                            Warning(initExpr->pos,
                                    "Initializer for global variable \"%s\" "
                                    "is a constant for single-target compilation "
                                    "but not for multi-target compilation.",
                                    name.c_str());
                        }

                        if (type->IsConstType()) {
                            // Try to get a ConstExpr associated with
                            // the symbol.  This llvm::dyn_cast can
                            // validly fail, for example for types like
                            // StructTypes where a ConstExpr can't
                            // represent their values.
                            constValue = llvm::dyn_cast<ConstExpr>(initExpr);
                        }
                    } else {
                        Error(initExpr->pos, "Initializer for global variable \"%s\" must be a constant.",
                              name.c_str());
                    }
                }
            }
        }

        // If no initializer was provided or if we couldn't get a value
        // above, initialize it with zeros..
        if (llvmInitializer == nullptr) {
            llvmInitializer = llvm::Constant::getNullValue(llvmType);
        }
    }

    Symbol *sym = symbolTable->LookupVariable(name.c_str());

    if (sym != nullptr) {
        // We've already seen either a declaration or a definition of this
        // global.

        // If the type doesn't match with the previous one, issue an error.
        if (!Type::Equal(sym->type, type) || (!sym->storageClass.IsAnyExtern() && sym->storageClass != storageClass)) {
            Error(pos, "Definition of variable \"%s\" conflicts with definition at %s:%d.", name.c_str(), sym->pos.name,
                  sym->pos.first_line);
            return;
        }

        llvm::GlobalVariable *gv = llvm::dyn_cast<llvm::GlobalVariable>(sym->storageInfo->getPointer());
        Assert(gv != nullptr);

        // And issue an error if this is a redefinition of a variable
        if (gv->hasInitializer() && !sym->storageClass.IsAnyExtern()) {
            Error(pos, "Redefinition of variable \"%s\" is illegal. (Previous definition at %s:%d.)", sym->name.c_str(),
                  sym->pos.name, sym->pos.first_line);
            return;
        }

        // Update existing global variable
        if (!gv->hasInitializer() && llvmInitializer) {
            gv->setInitializer(llvmInitializer);
        }

        // Update other properties
        gv->setConstant(isConst);

        // Apply alignment if specified
        if (alignment > 0) {
            gv->setAlignment(llvm::Align(alignment));
        }

        // Update the symbol's constValue
        sym->constValue = constValue;

        return;
    }
    sym = new Symbol(name, pos, Symbol::SymbolKind::Variable, type, storageClass);
    symbolTable->AddVariable(sym);

    sym->constValue = constValue;

    llvm::GlobalValue::LinkageTypes linkage =
        sym->storageClass.IsStatic() ? llvm::GlobalValue::InternalLinkage : llvm::GlobalValue::ExternalLinkage;

    // Note that the nullptr llvmInitializer is what leads to "extern"
    // declarations coming up extern and not defining storage (a bit
    // subtle)...
    // Create new GlobalVariable only for new symbols
    sym->storageInfo = new AddressInfo(
        new llvm::GlobalVariable(*module, llvmType, isConst, linkage, llvmInitializer, sym->name.c_str()), llvmType);

    // Apply alignment if specified
    if (alignment > 0) {
        llvm::GlobalVariable *gv = llvm::cast<llvm::GlobalVariable>(sym->storageInfo->getPointer());
        gv->setAlignment(llvm::Align(alignment));
    }

    if (diBuilder) {
        llvm::DIFile *file = pos.GetDIFile();
        llvm::DINamespace *diSpace = pos.GetDINamespace();
        // llvm::MDFile *file = pos.GetDIFile();
        llvm::GlobalVariable *sym_GV_storagePtr = llvm::dyn_cast<llvm::GlobalVariable>(sym->storageInfo->getPointer());
        Assert(sym_GV_storagePtr);
        llvm::DIGlobalVariableExpression *var = diBuilder->createGlobalVariableExpression(
            diSpace, name, name, file, pos.first_line, sym->type->GetDIType(diSpace), sym->storageClass.IsStatic());
        sym_GV_storagePtr->addDebugInfo(var);
    }
}

/** Given an arbitrary type, see if it or any of the leaf types contained
    in it has a type that's illegal to have exported to C/C++
    code.

    (Note that it's fine for the original struct or a contained struct to
    be varying, so long as all of its members have bound 'uniform'
    variability.)

    This functions returns true and issues an error if are any illegal
    types are found and returns false otherwise.
*/
static bool lRecursiveCheckValidParamType(const Type *t, bool vectorOk, bool soaOk, const std::string &name,
                                          SourcePos pos) {
    const StructType *st = CastType<StructType>(t);
    if (st != nullptr) {
        for (int i = 0; i < st->GetElementCount(); ++i) {
            if (!lRecursiveCheckValidParamType(st->GetElementType(i), soaOk, vectorOk, name, pos)) {
                return false;
            }
        }
        return true;
    }

    // Vector types are also not supported, pending ispc properly
    // supporting the platform ABI.  (Pointers to vector types are ok,
    // though.)  (https://github.com/ispc/ispc/issues/363)...
    if (vectorOk == false && CastType<VectorType>(t) != nullptr) {
        return false;
    }

    const SequentialType *seqt = CastType<SequentialType>(t);
    if (seqt != nullptr) {
        return lRecursiveCheckValidParamType(seqt->GetElementType(), soaOk, vectorOk, name, pos);
    }

    const PointerType *pt = CastType<PointerType>(t);
    if (pt != nullptr) {
        // Only allow exported uniform pointers
        // Uniform pointers to varying data, however, are ok.
        if (pt->IsVaryingType()) {
            return false;
        } else {
            return lRecursiveCheckValidParamType(pt->GetBaseType(), true, true, name, pos);
        }
    }

    if (t->IsSOAType() && soaOk) {
        Warning(pos, "Exported function parameter \"%s\" points to SOA type.", name.c_str());
        return false;
    }

    if (t->IsVaryingType() && !vectorOk) {
        return false;
    } else {
        if (t->IsVaryingType()) {
            Warning(pos, "Exported function parameter \"%s\" points to varying type", name.c_str());
        }
        return true;
    }
}

/** Given a Symbol representing a function parameter, see if it or any
    contained types are varying.  If so, issue an error.  (This function
    should only be called for parameters to 'export'ed functions, where
    varying parameters is illegal.
 */
static void lCheckExportedParameterTypes(const Type *type, const std::string &name, SourcePos pos) {
    if (lRecursiveCheckValidParamType(type, false, false, name, pos) == false) {
        if (const PointerType *pt = CastType<PointerType>(type)) {
            bool isSOAType = false;
            while (pt) {
                if (pt->GetBaseType()->IsSOAType()) {
                    isSOAType = true;
                    Error(pos, "SOA type parameter \"%s\" is illegal in an exported function.", name.c_str());
                }
                pt = CastType<PointerType>(pt->GetBaseType());
            }
            if (!isSOAType) {
                Error(pos, "Varying pointer type parameter \"%s\" is illegal in an exported function.", name.c_str());
            }
        }
        if (CastType<StructType>(type->GetBaseType())) {
            Error(pos, "Struct parameter \"%s\" with vector typed member(s) is illegal in an exported function.",
                  name.c_str());
        } else if (CastType<VectorType>(type)) {
            Error(pos, "Vector-typed parameter \"%s\" is illegal in an exported function.", name.c_str());
        } else {
            Error(pos, "Varying parameter \"%s\" is illegal in an exported function.", name.c_str());
        }
    }
}

#ifdef ISPC_XE_ENABLED
// For Xe target we have the same limitations in input parameters as for "export" functions
static void lCheckTaskParameterTypes(const Type *type, const std::string &name, SourcePos pos) {
    if (lRecursiveCheckValidParamType(type, false, false, name, pos) == false) {
        if (CastType<PointerType>(type))
            Error(pos, "Varying pointer type parameter \"%s\" is illegal in an \"task\" for Xe targets.", name.c_str());
        if (CastType<StructType>(type->GetBaseType()))
            Error(pos, "Struct parameter \"%s\" with vector typed member(s) is illegal in an \"task\" for Xe targets.",
                  name.c_str());
        else if (CastType<VectorType>(type))
            Error(pos,
                  "Vector-typed parameter \"%s\" is illegal in an \"task\" "
                  "for Xe targets.",
                  name.c_str());
        else
            Error(pos, "Varying parameter \"%s\" is illegal in an \"task\" for Xe targets.", name.c_str());
    }
}
#endif

/** Given a function type, loop through the function parameters and see if
    any are StructTypes.  If so, issue an error; this is currently broken
    (https://github.com/ispc/ispc/issues/3).
 */
static void lCheckForStructParameters(const FunctionType *ftype, SourcePos pos) {
    for (int i = 0; i < ftype->GetNumParameters(); ++i) {
        const Type *type = ftype->GetParameterType(i);
        if (CastType<StructType>(type) != nullptr) {
            Error(pos, "Passing structs to/from application functions by value "
                       "is currently not supported. Use a reference, a const reference, "
                       "a pointer, or a const pointer to the struct instead.");
            return;
        }
    }
}

/** We've got a declaration for a function to process.  This function does
    all the work of creating the corresponding llvm::Function instance,
    adding the symbol for the function to the symbol table and doing
    various sanity checks.
 */
void Module::AddFunctionDeclaration(const std::string &name, const FunctionType *functionType,
                                    StorageClass storageClass, Declarator *decl, bool isInline, bool isNoInline,
                                    bool isVectorCall, bool isRegCall, SourcePos pos) {
    Assert(functionType != nullptr);

    // If a global variable with the same name has already been declared
    // issue an error.
    if (symbolTable->LookupVariable(name.c_str()) != nullptr) {
        Error(pos, "Function \"%s\" shadows previously-declared global variable. Ignoring this definition.",
              name.c_str());
        return;
    }

    std::vector<Symbol *> overloadFuncs;
    symbolTable->LookupFunction(name.c_str(), &overloadFuncs);
    if (overloadFuncs.size() > 0) {
        for (unsigned int i = 0; i < overloadFuncs.size(); ++i) {
            Symbol *overloadFunc = overloadFuncs[i];

            const FunctionType *overloadType = CastType<FunctionType>(overloadFunc->type);
            if (overloadType == nullptr) {
                Assert(m->errorCount == 0);
                continue;
            }

            // Check for a redeclaration of a function with the same name
            // and type.  This also hits when we have previously declared
            // the function and are about to define it.
            if (Type::Equal(overloadFunc->type, functionType)) {
                return;
            }

            if (functionType->IsExported() || overloadType->IsExported()) {
                Error(pos,
                      "Illegal to provide \"export\" qualifier for "
                      "functions with the same name but different types. "
                      "(Previous function declaration (%s:%d).)",
                      overloadFunc->pos.name, overloadFunc->pos.first_line);
            }

            // If all of the parameter types match but the return type is
            // different, return an error--overloading by return type isn't
            // allowed.
            const FunctionType *ofType = CastType<FunctionType>(overloadFunc->type);
            Assert(ofType != nullptr);
            if (ofType->GetNumParameters() == functionType->GetNumParameters()) {
                int i = 0;
                for (i = 0; i < functionType->GetNumParameters(); ++i) {
                    if (Type::Equal(ofType->GetParameterType(i), functionType->GetParameterType(i)) == false) {
                        break;
                    }
                }
                if (i == functionType->GetNumParameters()) {
                    std::string thisRetType = functionType->GetReturnTypeString();
                    std::string otherRetType = ofType->GetReturnTypeString();
                    Error(pos,
                          "Illegal to overload function by return "
                          "type only.  This function returns \"%s\" while "
                          "previous declaration at %s:%d returns \"%s\".",
                          thisRetType.c_str(), overloadFunc->pos.name, overloadFunc->pos.first_line,
                          otherRetType.c_str());
                    return;
                }
            }
        }
    }

    bool isExternCorSYCL = storageClass.IsExternC() || storageClass.IsExternSYCL();
    if (isExternCorSYCL) {
        // Make sure the user hasn't supplied both an 'extern "C"' and a
        // 'task' qualifier with the function
        if (functionType->IsTask()) {
            Error(pos, "\"task\" qualifier is illegal with C-linkage extern function \"%s\".  Ignoring this function.",
                  name.c_str());
            return;
        }

        std::vector<Symbol *> funcs;
        symbolTable->LookupFunction(name.c_str(), &funcs);
        if (funcs.size() > 0) {
            if (funcs.size() > 1) {
                // Multiple functions with this name have already been declared;
                // can't overload here
                Error(pos,
                      "Can't overload extern \"C\" function \"%s\"; "
                      "%d functions with the same name have already been declared.",
                      name.c_str(), (int)funcs.size());
                return;
            }

            // One function with the same name has been declared; see if it
            // has the same type as this one, in which case it's ok.
            if (Type::Equal(funcs[0]->type, functionType)) {
                return;
            } else {
                Error(pos, "Can't overload extern \"C\" function \"%s\".", name.c_str());
                return;
            }
        }
    }

    // Get the LLVM FunctionType
    auto [name_pref, name_suf] = functionType->GetFunctionMangledName(false);
    std::string functionName = name_pref + name + name_suf;

    for (int i = 0; i < functionType->GetNumParameters(); i++) {
        const Type *t = functionType->GetParameterType(i);
        if (!t->IsCompleteType()) {
            const SourcePos paramPos = functionType->GetParameterSourcePos(i);
            const std::string paramName = functionType->GetParameterName(i);
            Error(paramPos, "parameter '%s' has incomplete type '%s'", paramName.c_str(), t->GetString().c_str());
            return;
        }
    }
    llvm::Function *function = functionType->CreateLLVMFunction(functionName, g->ctx, /*disableMask*/ isExternCorSYCL);

    if (g->target_os == TargetOS::windows) {
        // Make export functions callable from DLLs.
        if ((g->dllExport) && !storageClass.IsStatic()) {
            function->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
        }
    }

    if (isNoInline && isInline) {
        Error(pos, "Illegal to use \"noinline\" and \"inline\" qualifiers together on function \"%s\".", name.c_str());
        return;
    }
    // Set function attributes: we never throw exceptions
    function->setDoesNotThrow();
    if (!isExternCorSYCL && isInline) {
        function->addFnAttr(llvm::Attribute::AlwaysInline);
    }

    const bool isInternal = storageClass.IsStatic() || isInline;
    const bool isExternal = functionType->IsExported() || isExternCorSYCL;

    if (isInternal && isExternal) {
        const char *internalQualifier = isInline ? "inline" : "static";
        const char *externalQualifier = isExternCorSYCL ? "extern" : "export";
        Error(pos, "Function qualifier '%s' incompatible with '%s'.", externalQualifier, internalQualifier);
    }

    if (isVectorCall) {
        if (!storageClass.IsExternC()) {
            Error(pos, "Illegal to use \"__vectorcall\" qualifier on non-extern function \"%s\".", name.c_str());
            return;
        }
        if (g->target_os != TargetOS::windows) {
            Error(pos, "Illegal to use \"__vectorcall\" qualifier on function \"%s\" for non-Windows OS.",
                  name.c_str());
            return;
        }
    }

    if (isRegCall) {
        if (!isExternCorSYCL) {
            Error(pos, "Illegal to use \"__regcall\" qualifier on non-extern function \"%s\".", name.c_str());
            return;
        }
    }

    if (isNoInline) {
        function->addFnAttr(llvm::Attribute::NoInline);
    }

    AddUWTableFuncAttr(function);

    if (const auto &al = decl->attributeList) {
        if (al->HasAttribute("memory")) {
            const auto &memory = al->GetAttribute("memory")->arg.stringVal;
            if (memory == "none") {
                function->setDoesNotAccessMemory();
            } else if (memory == "read") {
                function->setOnlyReadsMemory();
            } else {
                Error(pos, "Unknown memory attribute \"%s\".", memory.c_str());
            }
        }
    }

    if (functionType->IsTask()) {
        if (!g->target->isXeTarget()) {
            // This also applies transitively to members I think?
            function->addParamAttr(0, llvm::Attribute::NoAlias);
        }
    }
    function->setCallingConv(functionType->GetCallingConv());
    g->target->markFuncWithTargetAttr(function);

    // Make sure that the return type isn't 'varying' or vector typed if
    // the function is 'export'ed.
    if (functionType->IsExported() &&
        lRecursiveCheckValidParamType(functionType->GetReturnType(), false, false, name, pos) == false) {
        Error(pos,
              "Illegal to return a \"varying\" or vector type from "
              "exported function \"%s\"",
              name.c_str());
    }

    if (functionType->IsTask() && functionType->GetReturnType()->IsVoidType() == false) {
        Error(pos, "Task-qualified functions must have void return type.");
    }

    if (functionType->IsExported() || functionType->IsExternC() || functionType->IsExternSYCL() ||
        functionType->IsISPCExternal() || functionType->IsISPCKernel()) {
        lCheckForStructParameters(functionType, pos);
    }

    // Mark with corresponding attribute
    if (g->target->isXeTarget()) {
        if (functionType->IsISPCKernel()) {
            function->addFnAttr("CMGenxMain");
        } else {
            function->addFnAttr("CMStackCall");
        }
    }

    // Loop over all of the arguments; process default values if present
    // and do other checks and parameter attribute setting.
    bool seenDefaultArg = false;
    int nArgs = functionType->GetNumParameters();
    for (int i = 0; i < nArgs; ++i) {
        const Type *argType = functionType->GetParameterType(i);
        const std::string &argName = functionType->GetParameterName(i);
        Expr *defaultValue = functionType->GetParameterDefault(i);
        const SourcePos &argPos = functionType->GetParameterSourcePos(i);

        // If the function is exported or in case of Xe target is task, make sure that the parameter
        // doesn't have any funky stuff going on in it.
        // JCB nomosoa - Varying is now a-ok.
        if (functionType->IsExported()) {
            lCheckExportedParameterTypes(argType, argName, argPos);
        }
#ifdef ISPC_XE_ENABLED
        if (functionType->IsISPCKernel()) {
            lCheckTaskParameterTypes(argType, argName, argPos);
        }
#endif

        // ISPC assumes that no pointers alias.  (It should be possible to
        // specify when this is not the case, but this should be the
        // default.)  Set parameter attributes accordingly.  (Only for
        // uniform pointers, since varying pointers are int vectors...)
        if (!functionType->IsTask() && !functionType->IsExternSYCL() &&
            ((CastType<PointerType>(argType) != nullptr && argType->IsUniformType() &&
              // Exclude SOA argument because it is a pair {struct *, int}
              // instead of pointer
              !CastType<PointerType>(argType)->IsSlice()) ||

             CastType<ReferenceType>(argType) != nullptr)) {

            function->addParamAttr(i, llvm::Attribute::NoAlias);
        }

        Assert(decl && decl->functionParams.size() == nArgs);
        DeclSpecs *declSpecs = decl->functionParams[i]->declSpecs;
        AttributeList *attrList = declSpecs ? declSpecs->attributeList : nullptr;
        if (attrList) {
            // Check for unknown attributes for parameters in function declarations.
            attrList->CheckForUnknownAttributes(decl->pos);

            if (attrList->HasAttribute("noescape")) {
                if (argType->IsPointerType() && argType->IsUniformType()) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_21_0
                    function->addParamAttr(
                        i, llvm::Attribute::getWithCaptureInfo(function->getContext(), llvm::CaptureInfo::none()));
#else
                    function->addParamAttr(i, llvm::Attribute::NoCapture);
#endif
                }

                if (argType->IsVaryingType()) {
                    Error(argPos, "\"noescape\" attribute illegal with \"varying\" parameter \"%s\".", argName.c_str());
                }

                if (!argType->IsPointerType()) {
                    Error(argPos, "\"noescape\" attribute illegal with non-pointer parameter \"%s\".", argName.c_str());
                }
            }
        }

        if (symbolTable->LookupFunction(argName.c_str())) {
            Warning(argPos, "Function parameter \"%s\" shadows a function declared in global scope.", argName.c_str());
        }

        if (symbolTable->LookupFunctionTemplate(argName.c_str())) {
            Warning(argPos, "Function parameter \"%s\" shadows a function template declared in global scope.",
                    argName.c_str());
        }

        if (defaultValue != nullptr) {
            seenDefaultArg = true;
        } else if (seenDefaultArg) {
            // Once one parameter has provided a default value, then all of
            // the following ones must have them as well.
            Error(argPos,
                  "Parameter \"%s\" is missing default: all "
                  "parameters after the first parameter with a default value "
                  "must have default values as well.",
                  argName.c_str());
        }
    }

    // If llvm gave us back a Function * with a different name than the one
    // we asked for, then there's already a function with that same
    // (mangled) name in the llvm::Module.  In that case, erase the one we
    // tried to add and just work with the one it already had.
    if (function->getName() != functionName) {
        function->eraseFromParent();
        function = module->getFunction(functionName);
    }

    // Finally, we know all is good and we can add the function to the
    // symbol table
    Symbol *funSym =
        new Symbol(name, pos, Symbol::SymbolKind::Function, functionType, storageClass, decl->attributeList);
    funSym->function = function;
    bool ok = symbolTable->AddFunction(funSym);
    Assert(ok);
}

void Module::AddFunctionDefinition(const std::string &name, const FunctionType *type, Stmt *code) {
    Symbol *sym = symbolTable->LookupFunction(name.c_str(), type);
    if (sym == nullptr || code == nullptr) {
        Assert(m->errorCount > 0);
        return;
    }

    sym->pos = code->pos;

    // FIXME: because we encode the parameter names in the function type,
    // we need to override the function type here in case the function had
    // earlier been declared with anonymous parameter names but is now
    // defined with actual names.  This is yet another reason we shouldn't
    // include the names in FunctionType...
    sym->type = type;

    ast->AddFunction(sym, code);
}

//
void Module::AddFunctionTemplateDeclaration(const TemplateParms *templateParmList, const std::string &name,
                                            const FunctionType *ftype, StorageClass sc, bool isInline, bool isNoInline,
                                            SourcePos pos) {
    Assert(ftype != nullptr);
    Assert(templateParmList != nullptr);

    // If a global variable with the same name has already been declared
    // issue an error.
    if (symbolTable->LookupVariable(name.c_str()) != nullptr) {
        Error(pos,
              "Function template \"%s\" shadows previously-declared global variable. "
              "Ignoring this definition.",
              name.c_str());
        return;
    }

    // Check overloads if the function template is already declared and we may skip
    // creating a new symbol.
    std::vector<TemplateSymbol *> overloadFuncTempls;
    bool foundAny = symbolTable->LookupFunctionTemplate(name, &overloadFuncTempls);

    if (foundAny) {
        for (unsigned int i = 0; i < overloadFuncTempls.size(); ++i) {
            TemplateSymbol *overloadFunc = overloadFuncTempls[i];

            const FunctionType *overloadType = overloadFunc->type;
            if (overloadType == nullptr) {
                Assert(m->errorCount == 0);
                continue;
            }

            // Check for a redeclaration of a function template with the same name
            // and type.  This also hits when we have previously declared
            // the function and are about to define it.
            if (templateParmList->IsEqual(overloadFunc->templateParms) && Type::Equal(overloadFunc->type, ftype)) {
                return;
            }
        }
    }

    // TODO: Xe adjust linkage

    // No mangling for template, only instantiations.

    // TODO: inline / noinline check.

    // ...

    // We don't need to worry here about passing vectorcall/regcall specifier since we support these
    // conventions for extern "C"/extern "SYCL" functions only. And we don't support extern "C"/extern "SYCL" functions
    // for templates. In case when --vectorcall option is passed on Windows, we will generate vectorcall functions for
    // all functions.
    TemplateSymbol *funcTemplSym = new TemplateSymbol(templateParmList, name, ftype, sc, pos, isInline, isNoInline);
    symbolTable->AddFunctionTemplate(funcTemplSym);
}

void Module::AddFunctionTemplateDefinition(const TemplateParms *templateParmList, const std::string &name,
                                           const FunctionType *ftype, Stmt *code) {
    if (templateParmList == nullptr || ftype == nullptr) {
        return;
    }

    TemplateSymbol *sym = symbolTable->LookupFunctionTemplate(templateParmList, name, ftype);
    if (sym == nullptr || code == nullptr) {
        Assert(m->errorCount > 0);
        return;
    }

    // Same trick, as in AddFunctionDefinition - update source position and the type to be the ones from function
    // template definition, not declaration.
    // This actually a hack, which addresses lack of expressiveness of AST, which doesn't represent pure declaration,
    // only definitions.
    sym->pos = code->pos;
    sym->type = ftype;

    ast->AddFunctionTemplate(sym, code);
}

FunctionTemplate *Module::MatchFunctionTemplate(const std::string &name, const FunctionType *ftype,
                                                TemplateArgs &normTypes, SourcePos pos) {
    if (ftype == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    std::vector<TemplateSymbol *> matches;
    bool found = symbolTable->LookupFunctionTemplate(name, &matches);
    if (!found) {
        Error(pos, "No matching function template was found.");
        return nullptr;
    }
    // Do template argument "normalization", i.e apply "varying type default":
    //
    // template <typename T> void foo(T t);
    // foo<int>(1); // T is assumed to be "varying int" here.
    for (auto &arg : normTypes) {
        arg.SetAsVaryingType();
    }

    FunctionTemplate *templ = nullptr;
    for (auto &templateSymbol : matches) {
        // Number of template parameters must match.
        if (normTypes.size() != templateSymbol->templateParms->GetCount()) {
            // We don't have default parameters yet, so just matching the size exactly.
            continue;
        }

        // Number of function parameters must match.
        if (!ftype || !templateSymbol->type || ftype->GetNumParameters() != templateSymbol->type->GetNumParameters()) {
            continue;
        }
        bool matched = true;
        TemplateInstantiation inst(*(templateSymbol->templateParms), normTypes, templateSymbol->isInline,
                                   templateSymbol->isNoInline);
        for (int i = 0; i < ftype->GetNumParameters(); i++) {
            const Type *instParam = ftype->GetParameterType(i);
            const Type *templateParam = templateSymbol->type->GetParameterType(i)->ResolveDependence(inst);
            if (!Type::Equal(instParam, templateParam)) {
                matched = false;
                break;
            }
        }
        if (matched) {
            templ = templateSymbol->functionTemplate;
            break;
        }
    }
    return templ;
}

void Module::AddFunctionTemplateInstantiation(const std::string &name, const TemplateArgs &tArgs,
                                              const FunctionType *ftype, StorageClass sc, bool isInline,
                                              bool isNoInline, SourcePos pos) {
    TemplateArgs normTypes(tArgs);
    FunctionTemplate *templ = MatchFunctionTemplate(name, ftype, normTypes, pos);
    if (templ) {
        // If primary template has default storage class, but explicit instantiation has non-default storage class,
        // report an error
        if (templ->GetStorageClass().IsNone() && !sc.IsNone()) {
            Error(pos, "Template instantiation has inconsistent storage class. Consider assigning it to the primary "
                       "template to inherit it's signature.");
            return;
        }
        // If primary template has non-default storage class, but explicit instantiation has different non-default
        // storage class, report an error
        if (!templ->GetStorageClass().IsNone() && !sc.IsNone() && sc != templ->GetStorageClass()) {
            Error(pos, "Template instantiation has inconsistent storage class.");
            return;
        }
        // If primary template doesn't have unmasked specifier, but explicit instantiation has it,
        // report an error
        if (!templ->GetFunctionType()->IsUnmasked() && ftype->IsUnmasked()) {
            Error(pos, "Template instantiation has inconsistent \"unmasked\" specifier. Consider moving the specifier "
                       "inside the function or assigning it to the primary template to inherit it's signature.");
            return;
        }
        templ->AddInstantiation(normTypes, TemplateInstantiationKind::Explicit, isInline, isNoInline);
    } else {
        Error(pos, "No matching function template found for instantiation.");
    }
}

void Module::AddFunctionTemplateSpecializationDefinition(const std::string &name, const FunctionType *ftype,
                                                         const TemplateArgs &tArgs, SourcePos pos, Stmt *code) {
    TemplateArgs normTypes(tArgs);
    FunctionTemplate *templ = MatchFunctionTemplate(name, ftype, normTypes, pos);
    if (templ == nullptr) {
        Error(pos, "No matching function template found for specialization.");
        return;
    }
    Symbol *sym = templ->LookupInstantiation(normTypes);
    if (sym == nullptr || code == nullptr) {
        Assert(m->errorCount > 0);
        return;
    }
    sym->pos = code->pos;
    // Update already created symbol with real function type and function implementation.
    // Inherit unmasked specifier from the basic template.
    const FunctionType *instType = CastType<FunctionType>(sym->type);
    bool instUnmasked = instType ? instType->IsUnmasked() : false;
    sym->type = instUnmasked ? ftype->GetAsUnmaskedType() : ftype->GetAsNonUnmaskedType();
    Function *inst = new Function(sym, code);
    sym->parentFunction = inst;
}

void Module::AddFunctionTemplateSpecializationDeclaration(const std::string &name, const FunctionType *ftype,
                                                          const TemplateArgs &tArgs, StorageClass sc, bool isInline,
                                                          bool isNoInline, SourcePos pos) {
    TemplateArgs normTypes(tArgs);
    FunctionTemplate *templ = MatchFunctionTemplate(name, ftype, normTypes, pos);
    if (templ == nullptr) {
        Error(pos, "No matching function template found for specialization.");
        return;
    }
    // If primary template has default storage class, but specialization has non-default storage class,
    // report an error
    if (templ->GetStorageClass().IsNone() && !sc.IsNone()) {
        Error(pos, "Template specialization has inconsistent storage class. Consider assigning it to the primary "
                   "template to inherit it's signature.");
        return;
    }
    // If primary template has non-default storage class, but specialization has different non-default storage class,
    // report an error
    if (!templ->GetStorageClass().IsNone() && !sc.IsNone() && sc != templ->GetStorageClass()) {
        Error(pos, "Template specialization has inconsistent storage class.");
        return;
    }
    // If primary template doesn't have unmasked specifier, but specialization has it,
    // report an error
    if (!templ->GetFunctionType()->IsUnmasked() && ftype->IsUnmasked()) {
        Error(pos, "Template specialization has inconsistent \"unmasked\" specifier. Consider moving the specifier "
                   "inside the function or assigning it to the primary template to inherit it's signature.");
        return;
    }
    Symbol *sym = templ->LookupInstantiation(normTypes);
    if (sym != nullptr) {
        if (Type::Equal(sym->type, ftype) && sym->parentFunction != nullptr) {
            Error(pos, "Template function specialization was already defined.");
            return;
        }
    }
    templ->AddSpecialization(ftype, normTypes, isInline, isNoInline, pos);
}

void Module::AddExportedTypes(const std::vector<std::pair<const Type *, SourcePos>> &types) {
    for (int i = 0; i < (int)types.size(); ++i) {
        if (CastType<StructType>(types[i].first) == nullptr && CastType<VectorType>(types[i].first) == nullptr &&
            CastType<EnumType>(types[i].first) == nullptr) {
            Error(types[i].second, "Only struct, vector, and enum types, not \"%s\", are allowed in type export lists.",
                  types[i].first->GetString().c_str());
        } else {
            exportedTypes.push_back(types[i]);
        }
    }
}

void Module::reportInvalidSuffixWarning(std::string filename, Module::OutputType outputType) {
    if (!filename.empty()) {
        // First, issue a warning if the output file suffix and the type of
        // file being created seem to mismatch.  This can help catch missing
        // command-line arguments specifying the output file type.
        std::size_t dotPos = filename.rfind('.');
        if (dotPos != std::string::npos) {
            std::string suffixStr = filename.substr(dotPos + 1);
            if (!(outputType >= 0 && outputType < outputTypeInfos.size())) {
                Assert(0 /* unhandled output type */);
            }

            const Module::OutputTypeInfo &info = outputTypeInfos[outputType];
            if (!info.isSuffixValid(suffixStr)) {
                Warning(SourcePos(), "Emitting %s file, but filename \"%s\" has suffix \"%s\"?", info.fileType,
                        filename.c_str(), suffixStr.c_str());
            }
        }
    }
}

bool Module::writeOutput() {
    OutputType outputType = output.type;

    // This function should not be called for generating outputs not related to
    // LLVM/Clang processing, i.e., header/deps/hostStub/devStub
    Assert(outputType != Header && outputType != Deps && outputType != HostStub && outputType != DevStub);
    Assert(module);

    // TODO: probably this is not good place to actually modify something inside the module
    if (diBuilder) {
        lStripUnusedDebugInfo(module);
    }

    if (g->functionSections) {
        module->addModuleFlag(llvm::Module::Warning, "function-sections", 1);
    }

    // In LLVM_3_4 after r195494 and r195504 revisions we should pass
    // "Debug Info Version" constant to the module. LLVM will ignore
    // our Debug Info metadata without it.
    if (g->generateDebuggingSymbols == true) {
        module->addModuleFlag(llvm::Module::Warning, "Debug Info Version", llvm::DEBUG_METADATA_VERSION);
    }

    // SIC! (verifyModule() == TRUE) means "failed", see llvm-link code.
    if ((outputType != CPPStub) && llvm::verifyModule(*module, &llvm::errs())) {
        FATAL("Resulting module verification failed!");
        return false;
    }

    reportInvalidSuffixWarning(output.out, outputType);

    switch (outputType) {
    case Asm:
    case Object:
        return writeObjectFileOrAssembly(module, output);
    case Bitcode:
    case BitcodeText:
        return writeBitcode(module, output.out, outputType);
    case CPPStub:
        return writeCPPStub();
#ifdef ISPC_XE_ENABLED
    case ZEBIN:
        return writeZEBin();
    case SPIRV:
        return writeSPIRV(module, output.out);
#endif
    // Do not process extra output types here better to call their methods directly
    case Header:
    case Deps:
    case DevStub:
    case HostStub:
    default:
        FATAL("Unhandled output type in Module::writeOutput()");
        return false;
    }
}

bool Module::writeBitcode(llvm::Module *module, std::string outFileName, OutputType outputType) {
    // Get a file descriptor corresponding to where we want the output to
    // go.  If we open it, it'll be closed by the llvm::raw_fd_ostream
    // destructor.
    int fd = -1;
    Assert(!outFileName.empty());
    if (outFileName == "-") {
        fd = 1; // stdout
    } else {
        int flags = O_CREAT | O_WRONLY | O_TRUNC;
#ifdef ISPC_HOST_IS_WINDOWS
        flags |= O_BINARY;
        fd = _open(outFileName.c_str(), flags, 0644);
#else
        fd = open(outFileName.c_str(), flags, 0644);
#endif // ISPC_HOST_IS_WINDOWS
        if (fd == -1) {
            perror(outFileName.c_str());
            return false;
        }
    }

    llvm::raw_fd_ostream fos(fd, (fd != 1), false);
    if (outputType == Bitcode) {
        llvm::WriteBitcodeToFile(*module, fos);
    } else if (outputType == BitcodeText) {
        module->print(fos, nullptr);
    }
    return true;
}

#ifdef ISPC_XE_ENABLED
static llvm::cl::opt<bool>
    SPIRVAllowExtraDIExpressions("spirv-allow-extra-diexpressions", llvm::cl::init(true),
                                 llvm::cl::desc("Allow DWARF operations not listed in the OpenCL.DebugInfo.100 "
                                                "specification (experimental, may produce incompatible SPIR-V "
                                                "module)"));

std::unique_ptr<llvm::Module> Module::translateFromSPIRV(std::ifstream &is) {
    std::string err;
    llvm::Module *m;
    SPIRV::TranslatorOpts Opts;
    Opts.enableAllExtensions();
    Opts.setAllowExtraDIExpressionsEnabled(SPIRVAllowExtraDIExpressions);
    Opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::SPIRVFriendlyIR); // the most important

    bool success = llvm::readSpirv(*g->ctx, Opts, is, m, err);
    if (!success) {
        fprintf(stderr, "Fails to read SPIR-V: %s \n", err.c_str());
        return nullptr;
    }
    return std::unique_ptr<llvm::Module>(m);
}

bool Module::translateToSPIRV(llvm::Module *module, std::stringstream &ss) {
    std::string err;
    SPIRV::TranslatorOpts Opts;
    Opts.enableAllExtensions();

    // At the moment, ocloc doesn't support SPV_KHR_untyped_pointers extension,
    // so turn it off. We may enable it later.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_20_0
    Opts.setAllowedToUseExtension(SPIRV::ExtensionID::SPV_KHR_untyped_pointers, false);
#endif

    Opts.setSPIRVAllowUnknownIntrinsics({"llvm.genx"});
    Opts.setAllowExtraDIExpressionsEnabled(SPIRVAllowExtraDIExpressions);
    Opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::SPIRVFriendlyIR);
    Opts.setDebugInfoEIS(SPIRV::DebugInfoEIS::OpenCL_DebugInfo_100);
    bool success = llvm::writeSpirv(module, Opts, ss, err);
    if (!success) {
        fprintf(stderr, "Fails to save LLVM as SPIR-V: %s \n", err.c_str());
        return false;
    }
    return true;
}

bool Module::writeSPIRV(llvm::Module *module, std::string outFileName) {
    std::stringstream translatedStream;
    bool success = translateToSPIRV(module, translatedStream);
    if (!success) {
        return false;
    }
    if (outFileName == "-") {
        std::cout << translatedStream.rdbuf();
    } else {
        std::ofstream fos(outFileName, std::ios::binary);
        fos << translatedStream.rdbuf();
    }
    return true;
}

// Copy outputs. Required file should have provided extension (it is
// different for different binary kinds).
static void saveOutput(uint32_t numOutputs, uint8_t **dataOutputs, uint64_t *lenOutputs, char **nameOutputs,
                       std::vector<char> &oclocRes) {
    const std::string &requiredExtension = ".bin";
    llvm::ArrayRef<const uint8_t *> outBins{dataOutputs, numOutputs};
    llvm::ArrayRef<uint64_t> outLens{lenOutputs, numOutputs};
    llvm::ArrayRef<const char *> outNames{nameOutputs, numOutputs};
    auto zip = llvm::zip(outBins, outLens, outNames);
    using ZipTy = typename decltype(zip)::value_type;
    auto binIt = std::find_if(zip.begin(), zip.end(), [&requiredExtension](ZipTy File) {
        llvm::StringRef name{std::get<2>(File)};
        return name.ends_with(requiredExtension);
    });
    Assert(binIt != zip.end() && "Output binary is missing");

    llvm::ArrayRef<uint8_t> binRef{std::get<0>(*binIt), static_cast<std::size_t>(std::get<1>(*binIt))};
    oclocRes.assign(binRef.begin(), binRef.end());
}

bool Module::writeZEBin() {
    std::stringstream translatedStream;
    bool success = translateToSPIRV(module, translatedStream);
    if (!success) {
        return false;
    }
    const std::string &translatedStr = translatedStream.str();
    std::vector<char> spirStr(translatedStr.begin(), translatedStr.end());

    const std::string CPUName = g->target->getCPU();

    invokePtr invoke;
    freeOutputPtr freeOutput;
    auto oclocLib = llvm::sys::DynamicLibrary::getPermanentLibrary(OCLOC_LIBRARY_NAME);
    if (!oclocLib.isValid()) {
        Error(SourcePos(), "Cannot open '" OCLOC_LIBRARY_NAME "'\n");
        return false;
    }
    invoke = reinterpret_cast<invokePtr>(oclocLib.getAddressOfSymbol("oclocInvoke"));
    if (!invoke) {
        Error(SourcePos(), "oclocInvoke symbol is missing \n");
        return false;
    }
    freeOutput = reinterpret_cast<freeOutputPtr>(oclocLib.getAddressOfSymbol("oclocFreeOutput"));
    if (!freeOutput) {
        Error(SourcePos(), "oclocFreeOutput symbol is missing \n");
        return false;
    }

    std::string options{"-vc-codegen -no-optimize -Xfinalizer '-presched' -Xfinalizer '-newspillcostispc'"};
    // Add debug option to VC backend of "-g" is set to ISPC
    if (g->generateDebuggingSymbols) {
        options.append(" -g");
    }

    if (g->vcOpts != "") {
        options.append(" " + g->vcOpts);
    }

    // Add stack size info
    // If stackMemSize has default value 0, do not set -stateless-stack-mem-size,
    // it will be set to 8192 in VC backend by default.
    if (g->stackMemSize > 0) {
        options.append(" -stateless-stack-mem-size=" + std::to_string(g->stackMemSize));
    }
    std::string internalOptions;

    // Use L0 binary
    internalOptions.append(" -binary-format=").append("ze");

    const char *spvFileName = "ispc_spirv";

    std::vector<const char *> oclocArgs;
    oclocArgs.push_back("ocloc");
    oclocArgs.push_back("compile");
    oclocArgs.push_back("-device");
    oclocArgs.push_back(CPUName.c_str());
    oclocArgs.push_back("-spirv_input");
    oclocArgs.push_back("-file");
    oclocArgs.push_back(spvFileName);
    oclocArgs.push_back("-options");
    oclocArgs.push_back(options.c_str());
    oclocArgs.push_back("-internal_options");
    oclocArgs.push_back(internalOptions.c_str());

    uint32_t numOutputs = 0;
    uint8_t **dataOutputs = nullptr;
    uint64_t *lenOutputs = nullptr;
    char **nameOutputs = nullptr;

    static_assert(alignof(uint8_t) == alignof(char), "Possible unaligned access");
    auto *spvSource = reinterpret_cast<const uint8_t *>(spirStr.data());
    const uint64_t spvLen = spirStr.size();
    if (invoke(oclocArgs.size(), oclocArgs.data(), 1, &spvSource, &spvLen, &spvFileName, 0, nullptr, nullptr, nullptr,
               &numOutputs, &dataOutputs, &lenOutputs, &nameOutputs)) {
        Error(SourcePos(), "Call to oclocInvoke failed \n");
        return false;
    }

    std::vector<char> oclocRes;
    saveOutput(numOutputs, dataOutputs, lenOutputs, nameOutputs, oclocRes);
    if (freeOutput(&numOutputs, &dataOutputs, &lenOutputs, &nameOutputs)) {
        Error(SourcePos(), "Call to oclocFreeOutput failed \n");
        return false;
    }

    if (output.out == "-") {
        std::cout.write(oclocRes.data(), oclocRes.size());
    } else {
        std::ofstream fos(output.out, std::ios::binary);
        fos.write(oclocRes.data(), oclocRes.size());
    }
    return true;
}
#endif // ISPC_XE_ENABLED

bool Module::writeObjectFileOrAssembly(llvm::Module *M, Output &CO) {
    llvm::TargetMachine *targetMachine = g->target->GetTargetMachine();
    Assert(targetMachine);

    // Figure out if we're generating object file or assembly output, and
    // set binary output for object files
#if ISPC_LLVM_VERSION > ISPC_LLVM_17_0
    llvm::CodeGenFileType fileType =
        (CO.type == Object) ? llvm::CodeGenFileType::ObjectFile : llvm::CodeGenFileType::AssemblyFile;
    bool binary = (fileType == llvm::CodeGenFileType::ObjectFile);
#else
    llvm::CodeGenFileType fileType = (CO.type == Object) ? llvm::CGFT_ObjectFile : llvm::CGFT_AssemblyFile;
    bool binary = (fileType == llvm::CGFT_ObjectFile);

#endif
    llvm::sys::fs::OpenFlags flags = binary ? llvm::sys::fs::OF_None : llvm::sys::fs::OF_Text;

    std::error_code error;

    std::unique_ptr<llvm::ToolOutputFile> of(new llvm::ToolOutputFile(CO.out, error, flags));

    if (error) {
        Error(SourcePos(), "Cannot open output file \"%s\".\n", CO.out.c_str());
        return false;
    }

    llvm::legacy::PassManager pm;

    {
        llvm::raw_fd_ostream &fos(of->os());
        // Third parameter is for generation of .dwo file, which is separate DWARF
        // file for ELF targets. We don't support it currently.
        if (targetMachine->addPassesToEmitFile(pm, fos, nullptr, fileType)) {
            FATAL("Failed to add passes to emit object file!");
        }

        // Finally, run the passes to emit the object file/assembly
        pm.run(*M);

        // Success; tell tool_output_file to keep the final output file.
        of->keep();
    }
    return true;
}

// Given an output filename of the form "foo.obj", and a Target
// return a string with the ISA name inserted before the original
// filename's suffix, like "foo_avx.obj".
std::string lGetMangledFileName(std::string filename, Target *target) {
    std::string isaString = target->GetISAString();

    assert(!filename.empty() && "`filename` should not be empty");
    std::string targetFileName{filename};
    const auto pos_dot = targetFileName.find_last_of('.');

    if (pos_dot != std::string::npos) {
        // If filename has an extension, insert ISA name before the dot
        targetFileName = targetFileName.substr(0, pos_dot) + "_" + isaString + targetFileName.substr(pos_dot);
    } else {
        // If no extension, simply append ISA name
        targetFileName.append("_" + isaString);
    }
    return targetFileName;
}

std::string Module::Output::OutFileNameTarget(Target *target) const { return lGetMangledFileName(out, target); }

std::string Module::Output::HeaderFileNameTarget(Target *target) const { return lGetMangledFileName(header, target); }

static bool lSymbolIsExported(const Symbol *s) { return s->exportedFunction != nullptr; }

// Small structure to hold pointers to the various different versions of a
// llvm::Function that were compiled for different compilation target ISAs.
struct FunctionTargetVariants {
    FunctionTargetVariants() {
        // TODO!: NUM_ISAS counts all ISAa, not only x86 ones.
        // Whereas, all these are x86 specific, because dispatch code is
        // generated only for x86.
        for (int i = 0; i < Target::NUM_ISAS; ++i) {
            func[i] = nullptr;
            FTs[i] = nullptr;
        }
    }
    // The func array is indexed with the Target::ISA enumerant.  Some
    // values may be nullptr, indicating that the original function wasn't
    // compiled to the corresponding target ISA.
    llvm::Function *func[Target::NUM_ISAS];
    const FunctionType *FTs[Target::NUM_ISAS];
};

// Given the symbol table for a module, return a map from function names to
// FunctionTargetVariants for each function that was defined with the
// 'export' qualifier in ispc.
static void lGetExportedFunctions(SymbolTable *symbolTable, std::map<std::string, FunctionTargetVariants> &functions) {
    std::vector<Symbol *> syms;
    symbolTable->GetMatchingFunctions(lSymbolIsExported, &syms);
    for (unsigned int i = 0; i < syms.size(); ++i) {
        FunctionTargetVariants &ftv = functions[syms[i]->name];
        ftv.func[g->target->getISA()] = syms[i]->exportedFunction;
        ftv.FTs[g->target->getISA()] = CastType<FunctionType>(syms[i]->type);
    }
}

static llvm::FunctionType *lGetVaryingDispatchType(FunctionTargetVariants &funcs) {
    llvm::FunctionType *resultFuncTy = nullptr;

    for (int i = 0; i < Target::NUM_ISAS; ++i) {
        if (funcs.func[i] == nullptr) {
            continue;
        } else {
            bool foundVarying = false;
            const FunctionType *ft = funcs.FTs[i];
            resultFuncTy = funcs.func[i]->getFunctionType();

            int numArgs = ft->GetNumParameters();
            llvm::SmallVector<llvm::Type *, 8> ftype;
            for (int j = 0; j < numArgs; ++j) {
                ftype.push_back(resultFuncTy->getParamType(j));
            }

            for (int j = 0; j < numArgs; ++j) {
                const Type *arg = ft->GetParameterType(j);

                if (arg->IsPointerType()) {
                    const Type *baseType = CastType<PointerType>(arg)->GetBaseType();
                    // For each varying type pointed to, swap the LLVM pointer type
                    // with i8 * (as close as we can get to void *)
                    if (baseType->IsVaryingType()) {
                        ftype[j] = LLVMTypes::VoidPointerType;
                        foundVarying = true;
                    }
                }
            }
            if (foundVarying) {
                resultFuncTy = llvm::FunctionType::get(resultFuncTy->getReturnType(), ftype, false);
            }
        }
    }

    // We should've found at least one variant here
    // or else something fishy is going on.
    Assert(resultFuncTy);

    return resultFuncTy;
}

/** Create the dispatch function for an exported ispc function.
    This function checks to see which vector ISAs the system the
    code is running on supports and calls out to the best available
    variant that was generated at compile time.

    @param module      Module in which to create the dispatch function.
    @param setISAFunc  Pointer to the __set_system_isa() function defined
                       in builtins-dispatch.ll (which is linked into the
                       given module before we get here.)
    @param systemBestISAPtr  Pointer to the module-local __system_best_isa
                             variable, which holds a value of the Target::ISA
                             enumerant giving the most capable ISA that the
                             system supports.
    @param name        Name of the function for which we're generating a
                       dispatch function
    @param funcs       Target-specific variants of the exported function.
*/
static void lCreateDispatchFunction(llvm::Module *module, llvm::Function *setISAFunc, llvm::Value *systemBestISAPtr,
                                    const std::string &name, FunctionTargetVariants &funcs) {
    // The llvm::Function pointers in funcs are pointers to functions in
    // different llvm::Modules, so we can't call them directly.  Therefore,
    // we'll start by generating an 'extern' declaration of each one that
    // we have in the current module so that we can then call out to that.
    llvm::Function *targetFuncs[Target::NUM_ISAS];

    // New helper function checks to see if we need to rewrite the
    // type for the dispatch function in case of pointers to varyings
    llvm::FunctionType *ftype = lGetVaryingDispatchType(funcs);

    // Now we insert type-punned declarations for dispatched functions.
    // This is needed when compiling modules for a set of architectures
    // with different vector lengths. Due to restrictions, the return
    // type is the same across all architectures, however in different
    // modules it may have dissimilar names. The loop below works this
    // around.
    unsigned int callingConv = llvm::CallingConv::C;
    for (int i = 0; i < Target::NUM_ISAS; ++i) {
        if (funcs.func[i]) {

            targetFuncs[i] =
                llvm::Function::Create(ftype, llvm::GlobalValue::ExternalLinkage, funcs.func[i]->getName(), module);
            // Calling convention should be the same for all dispatched functions
            callingConv = funcs.FTs[i]->GetCallingConv();
            targetFuncs[i]->setCallingConv(callingConv);
            AddUWTableFuncAttr(targetFuncs[i]);
        } else {
            targetFuncs[i] = nullptr;
        }
    }

    bool voidReturn = ftype->getReturnType()->isVoidTy();

    std::string functionName = name;
    if (callingConv == llvm::CallingConv::X86_RegCall) {
        g->target->markFuncNameWithRegCallPrefix(functionName);
    }

    // Now we can emit the definition of the dispatch function..
    llvm::Function *dispatchFunc =
        llvm::Function::Create(ftype, llvm::GlobalValue::ExternalLinkage, functionName.c_str(), module);
    dispatchFunc->setCallingConv(callingConv);
    AddUWTableFuncAttr(dispatchFunc);

    // Make dispatch function callable from DLLs.
    if ((g->target_os == TargetOS::windows) && (g->dllExport)) {
        dispatchFunc->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
    }
    llvm::BasicBlock *bblock = llvm::BasicBlock::Create(*g->ctx, "entry", dispatchFunc);

    // Start by calling out to the function that determines the system's
    // ISA and sets __system_best_isa, if it hasn't been set yet.
    llvm::CallInst::Create(setISAFunc, "", bblock);

    // Now we can load the system's ISA enumerant
    llvm::Value *systemISA = new llvm::LoadInst(LLVMTypes::Int32Type, systemBestISAPtr, "system_isa", bblock);

    // Now emit code that works backwards though the available variants of
    // the function.  We'll call out to the first one we find that will run
    // successfully on the system the code is running on.  In working
    // through the candidate ISAs here backward, we're taking advantage of
    // the expectation that they are ordered in the Target::ISA enumerant
    // from least to most capable.
    for (int i = Target::NUM_ISAS - 1; i >= 0; --i) {
        if (targetFuncs[i] == nullptr) {
            continue;
        }

        // Emit code to see if the system can run the current candidate
        // variant successfully--"is the system's ISA enumerant value >=
        // the enumerant value of the current candidate?"

        llvm::Value *ok = llvm::CmpInst::Create(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_SGE, systemISA,
                                                LLVMInt32(i), "isa_ok", bblock);
        llvm::BasicBlock *callBBlock = llvm::BasicBlock::Create(*g->ctx, "do_call", dispatchFunc);
        llvm::BasicBlock *nextBBlock = llvm::BasicBlock::Create(*g->ctx, "next_try", dispatchFunc);
        llvm::BranchInst::Create(callBBlock, nextBBlock, ok, bblock);

        // Emit the code to make the call call in callBBlock.
        // Just pass through all of the args from the dispatch function to
        // the target-specific function.
        std::vector<llvm::Value *> args;
        llvm::Function::arg_iterator argIter = dispatchFunc->arg_begin();
        llvm::Function::arg_iterator targsIter = targetFuncs[i]->arg_begin();
        for (; argIter != dispatchFunc->arg_end(); ++argIter, ++targsIter) {
            // Check to see if we rewrote any types in the dispatch function.
            // If so, create bitcasts for the appropriate pointer types.
            if (argIter->getType() == targsIter->getType()) {
                args.push_back(&*argIter);
            } else {
                llvm::CastInst *argCast = llvm::CastInst::CreatePointerCast(&*argIter, targsIter->getType(),
                                                                            "dpatch_arg_bitcast", callBBlock);
                args.push_back(argCast);
            }
        }
        if (voidReturn) {
            llvm::CallInst *callInst = llvm::CallInst::Create(targetFuncs[i], args, "", callBBlock);
            callInst->setCallingConv(targetFuncs[i]->getCallingConv());
            llvm::ReturnInst::Create(*g->ctx, callBBlock);
        } else {
            llvm::CallInst *callInst = llvm::CallInst::Create(targetFuncs[i], args, "ret_value", callBBlock);
            callInst->setCallingConv(targetFuncs[i]->getCallingConv());
            llvm::ReturnInst::Create(*g->ctx, callInst, callBBlock);
        }

        // Otherwise we'll go on to the next candidate and see about that
        // one...
        bblock = nextBBlock;
    }

    // We couldn't find a match that the current system was capable of running.
    // We used to call abort() here but replaced it with call to our own
    // implementation __terminate_now from builtins/dispatch.c
    // It might be preferable to call a user-supplied callback--ISPCError(...)
    // or some such, but we don't want to start imposing too much of a
    // runtime library requirement either...
    llvm::Function *abortFunc = module->getFunction(builtin::__terminate_now);
    Assert(abortFunc);
    llvm::CallInst::Create(abortFunc, "", bblock);

    // Return an undef value from the function here; we won't get to this
    // point at runtime, but LLVM needs all of the basic blocks to be
    // terminated...
    if (voidReturn) {
        llvm::ReturnInst::Create(*g->ctx, bblock);
    } else {
        llvm::Value *undefRet = llvm::UndefValue::get(ftype->getReturnType());
        llvm::ReturnInst::Create(*g->ctx, undefRet, bblock);
    }
}

// Initialize a dispatch module
static llvm::Module *lInitDispatchModule() {
    llvm::Module *module = new llvm::Module("dispatch_module", *g->ctx);

#if ISPC_LLVM_VERSION >= ISPC_LLVM_21_0
    module->setTargetTriple(g->target->GetTriple());
#else
    module->setTargetTriple(g->target->GetTriple().str());
#endif

    // DataLayout information supposed to be managed in single place in Target class.
    module->setDataLayout(g->target->getDataLayout()->getStringRepresentation());

    lSetCodeModel(module);
    lSetPICLevel(module);

#if ISPC_LLVM_VERSION >= ISPC_LLVM_19_0 && ISPC_LLVM_VERSION < ISPC_LLVM_21_0
    // LLVM is transitioning to new debug info representation, use "old" style for now.
    module->setIsNewDbgInfoFormat(false);
#endif

    if (!g->genStdlib) {
        // First, link in the definitions from the builtins-dispatch.ll file.
        LinkDispatcher(module);
    }

    return module;
}

// Complete the creation of a dispatch module.
// Given a map that holds the mapping from each of the 'export'ed functions
// in the ispc program to the target-specific variants of the function,
// create a llvm::Module that has a dispatch function for each exported
// function that checks the system's capabilities and picks the most
// appropriate compiled variant of the function.
static void lEmitDispatchModule(llvm::Module *module, std::map<std::string, FunctionTargetVariants> &functions) {
    // Get pointers to things we need below
    llvm::Function *setFunc = module->getFunction(builtin::__set_system_isa);
    Assert(setFunc != nullptr);
    llvm::Value *systemBestISAPtr = module->getGlobalVariable("__system_best_isa", true);
    Assert(systemBestISAPtr != nullptr);

    // For each exported function, create the dispatch function
    std::map<std::string, FunctionTargetVariants>::iterator iter;
    for (iter = functions.begin(); iter != functions.end(); ++iter) {
        lCreateDispatchFunction(module, setFunc, systemBestISAPtr, iter->first, iter->second);
    }

    // Do some rudimentary cleanup of the final result and make sure that
    // the module is all ok.
    llvm::ModulePassManager mpm;
    llvm::ModuleAnalysisManager mam;
    llvm::PassBuilder pb = llvm::PassBuilder();
    pb.registerModuleAnalyses(mam);
    mpm.addPass(llvm::GlobalDCEPass());
    mpm.addPass(llvm::VerifierPass());
    mpm.run(*module, mam);
}

// Determines if two types are compatible.
// Here we check layout compatibility. We don't care about pointer types.
static bool lCompatibleTypes(llvm::Type *Ty1, llvm::Type *Ty2) {
    while (Ty1->getTypeID() == Ty2->getTypeID()) {
        switch (Ty1->getTypeID()) {
        case llvm::Type::ArrayTyID:
            if (Ty1->getArrayNumElements() != Ty2->getArrayNumElements()) {
                return false;
            }
            Ty1 = Ty1->getArrayElementType();
            Ty2 = Ty2->getArrayElementType();
            break;
        case llvm::Type::PointerTyID:
            // Uniform pointers are compatible.
            // Varying pointers are represented as <N x i32>/<N x i64>, it'll
            // be checked in default statement.
            return true;
        case llvm::Type::StructTyID: {
            llvm::StructType *STy1 = llvm::dyn_cast<llvm::StructType>(Ty1);
            llvm::StructType *STy2 = llvm::dyn_cast<llvm::StructType>(Ty2);
            return STy1 && STy2 && STy1->isLayoutIdentical(STy2);
        }
        default:
            // Pointers for compatible simple types are assumed equal
            return Ty1 == Ty2;
        }
    }
    return false;
}

/**
 * Extract or validate global variables across multiple target modules during
 * multi-target compilation.
 *
 * This function handles two critical scenarios in multi-target compilation:
 * 1. Extraction mode (check = false):
 *    - Creates new global variable declarations in the destination module
 *    - Copies initializers from the source module
 *    - Preserves global variable attributes
 *
 * 2. Validation mode (check = true):
 *    - Verifies that global variables exist in the destination module
 *    - Checks type and layout compatibility across different target modules
 *    - Warns about potential mismatches in global variable definitions
 *
 * @param msrc Source module containing global variables to extract/check
 * @param mdst Destination module to either create or validate global variables
 * @param check Flag to determine whether to extract (false) or validate (true)
 */
static void lExtractOrCheckGlobals(llvm::Module *msrc, llvm::Module *mdst, bool check) {
    llvm::Module::global_iterator iter;
    llvm::ValueToValueMapTy VMap;

    for (iter = msrc->global_begin(); iter != msrc->global_end(); ++iter) {
        llvm::GlobalVariable *gv = &*iter;
        // Is it a global definition?
        if (gv->getLinkage() == llvm::GlobalValue::ExternalLinkage && gv->hasInitializer()) {
            llvm::Type *type = gv->getValueType();
            Symbol *sym = m->symbolTable->LookupVariable(gv->getName().str().c_str());
            Assert(sym != nullptr);

            // Check presence and compatibility for the current global
            if (check) {
                llvm::GlobalVariable *exist = mdst->getGlobalVariable(gv->getName());
                Assert(exist != nullptr);

                // It is possible that the types may not match: for
                // example, this happens with varying globals if we
                // compile to different vector widths
                if (!lCompatibleTypes(exist->getValueType(), gv->getValueType())) {
                    Warning(sym->pos,
                            "Mismatch in size/layout of global "
                            "variable \"%s\" with different targets. "
                            "Globals must not include \"varying\" types or arrays "
                            "with size based on programCount when compiling to "
                            "targets with differing vector widths.",
                            gv->getName().str().c_str());
                }
            }
            // Alternatively, create it anew and make it match the original
            else {
                llvm::GlobalVariable *newGlobal =
                    new llvm::GlobalVariable(*mdst, type, gv->isConstant(), llvm::GlobalValue::ExternalLinkage,
                                             (llvm::Constant *)nullptr, gv->getName());
                VMap[&*iter] = newGlobal;

                newGlobal->setInitializer(llvm::MapValue(iter->getInitializer(), VMap));
                newGlobal->copyAttributesFrom(gv);
            }
        }
    }
}

using InitializerStorage = std::map<llvm::GlobalVariable *, llvm::Constant *>;

/**
 * Temporarily removes initializers from global variables in the module.
 *
 * When compiling for multiple targets, we need to avoid duplicate definitions
 * of global variables with initializers. This function saves the current
 * initializers in the provided storage and then sets all global variables
 * to have no initializer, effectively turning them into external declarations.
 * The initializers can later be restored using restoreGlobalInitializers().
 *
 * @param module The LLVM module containing globals to process
 * @param initializers Storage map to save the current initializers
 */
void resetGlobalInitializers(llvm::Module *module, InitializerStorage &initializers) {
    for (llvm::GlobalVariable &gv : module->globals()) {
        if (gv.getLinkage() == llvm::GlobalVariable::ExternalLinkage && gv.hasInitializer()) {
            initializers[&gv] = gv.getInitializer();
            gv.setInitializer(nullptr);
        }
    }
}

/**
 * Restores previously saved initializers to global variables.
 *
 * This function reverses the effect of resetGlobalInitializers() by
 * restoring the saved initializers back to their global variables.
 * It's typically used after writing output files in multi-target mode.
 *
 * @param module The LLVM module containing globals to restore
 * @param initializers Storage map containing the saved initializers
 */
void restoreGlobalInitializers(llvm::Module *module, InitializerStorage &initializers) {
    for (llvm::GlobalVariable &gv : module->globals()) {
        if (gv.getLinkage() == llvm::GlobalVariable::ExternalLinkage && initializers.count(&gv)) {
            // Turn this into an 'extern' declaration by clearing its
            // initializer.
            gv.setInitializer(initializers[&gv]);
        }
    }
}

// For Xe targets, validate that the requested output format is supported
int lValidateXeTargetOutputType(Target *target, Module::OutputType &outputType) {
#ifdef ISPC_XE_ENABLED
    if (outputType == Module::OutputType::Asm || outputType == Module::OutputType::Object) {
        if (g->target->isXeTarget()) {
            Error(SourcePos(), "%s output is not supported yet for Xe targets. ",
                  (outputType == Module::OutputType::Asm) ? "assembly" : "binary");
            return 1;
        }
    }
    if (g->target->isXeTarget() && outputType == Module::OutputType::Object) {
        outputType = Module::OutputType::ZEBIN;
    }
    if (!g->target->isXeTarget() &&
        (outputType == Module::OutputType::ZEBIN || outputType == Module::OutputType::SPIRV)) {
        Error(SourcePos(), "SPIR-V and L0 binary formats are supported for Xe target only");
        return 1;
    }
#endif
    return 0;
}

/**
 * This function handles writing all requested output files for the current module,
 * including the main output file (object, assembly, bitcode), header file,
 * dependency information, and stub files for offload compilation.
 */
int Module::WriteOutputFiles() {
    // Write the main output file
    if (!output.out.empty() && !writeOutput()) {
        return 1;
    }
    if (!output.header.empty() && !writeHeader()) {
        return 1;
    }
    if (!output.nbWrap.empty() && !writeNanobindWrapper()) {
        return 1;
    }
    if (!output.deps.empty() || output.flags.isDepsToStdout()) {
        if (!writeDeps(output)) {
            return 1;
        }
    }
    if (!output.hostStub.empty() && !writeHostStub()) {
        return 1;
    }
    if (!output.devStub.empty() && !writeDevStub()) {
        return 1;
    }
    return 0;
}

/**
 * Compiles the module for a single target architecture.
 *
 * This function orchestrates the complete compilation process for a specific target,
 * beginning with parsing the source file, followed by code generation,
 * optimization, and output file creation.
 *
 * The compilation mode (Single vs Multiple) affects how global initializers
 * are handled. In multi-target mode, each target gets its own Module instance.
 *
 * In multi-target mode, globals and functions have different handling mechanisms:
 * - Global variables: Only declarations (externals) appear in target-specific modules.
 *   The actual definitions with initializers will appear only in the dispatch module.
 *   This function temporarily removes initializers before writing output files, making
 *   them external declarations, then restores them afterward so they can be extracted
 *   by lExtractOrCheckGlobals() for use in the dispatch module.
 *
 * - Functions: Target modules contain full definitions with target-specific mangled names
 *   (e.g., foo_sse4, foo_avx2). The dispatch module creates wrapper functions with the
 *   original names (e.g., foo) that dispatch to the appropriate target-specific implementation
 *   based on runtime CPU detection.
 *
 * @note For multi-target compilation, separate Module instances are created
 *       for each target, and this function is called once per Module.
 */
int Module::CompileSingleTarget(Arch arch, const char *cpu, ISPCTarget target) {
    // Compile the file by parsing source, generating IR, and applying optimizations
    const int compileResult = CompileFile();

    if (compileResult == 0) {
        llvm::TimeTraceScope TimeScope("Backend");

        if (lValidateXeTargetOutputType(g->target, output.type)) {
            return 1;
        }

        InitializerStorage initializers;
        if (m_compilationMode == CompilationMode::Multiple) {
            // For multi-target compilation, temporarily remove initializers
            resetGlobalInitializers(module, initializers);
        }

        // Generate the requested output files (object, assembly, bitcode, etc.)
        if (WriteOutputFiles()) {
            return 1;
        }

        if (m_compilationMode == CompilationMode::Multiple) {
            // Restore the initializers after writing output files.
            // This ensures they're available at the moment we generate the
            // dispatch module
            restoreGlobalInitializers(module, initializers);
        }
    } else {
        ++errorCount;

        // In case of error, clean up symbolTable
        symbolTable->PopInnerScopes();
    }

    return errorCount;
}

int lValidateMultiTargetInputs(const char *srcFile, std::string &outFileName, const char *cpu) {
    if (IsStdin(srcFile)) {
        Error(SourcePos(), "Compiling programs from standard input isn't "
                           "supported when compiling for multiple targets.  Please use "
                           "an intermediate temporary file.");
        return 1;
    }
    if (cpu != nullptr) {
        Error(SourcePos(), "Illegal to specify cpu type when compiling for multiple targets.");
        return 1;
    }
    if (outFileName == "-") {
        Error(SourcePos(), "Multi-target compilation can't generate output "
                           "to stdout.  Please provide an output filename.\n");
        return 1;
    }
    return 0;
}

// After generating the dispatch module, this function handles writing it to
// disk in the requested format (object file, assembly, bitcode, etc.) and
// generates any requested supplementary files like dependencies.
int Module::WriteDispatchOutputFiles(llvm::Module *dispatchModule, Module::Output &output) {
    if (!output.out.empty()) {
        switch (output.type) {
        case Module::OutputType::CPPStub:
            // No preprocessor output for dispatch module.
            break;

        case Module::OutputType::Bitcode:
        case Module::OutputType::BitcodeText:
            if (!m->writeBitcode(dispatchModule, output.out.c_str(), output.type)) {
                return 1;
            }
            break;

        case Module::OutputType::Asm:
        case Module::OutputType::Object:
            if (!m->writeObjectFileOrAssembly(dispatchModule, output)) {
                return 1;
            }
            break;

        default:
            FATAL("Unexpected `outputType`");
        }
    }

    if (!output.deps.empty() || output.flags.isDepsToStdout()) {
        if (!m->writeDeps(output)) {
            return 1;
        }
    }

    return 0;
}

std::pair<Target::ISA, int> lCheckAndFillISAIndices(std::vector<ISPCTarget> targets) {
    std::map<Target::ISA, int> ISAIndices;

    for (unsigned int i = 0; i < targets.size(); ++i) {
        auto ISA = Target::TargetToISA(targets[i]);

        // Issue an error if we've already compiled to a variant of
        // this target ISA.  (It doesn't make sense to compile to both
        // avx and avx-x2, for example.)
        if (ISAIndices.find(ISA) != ISAIndices.end()) {
            Error(SourcePos(), "Can't compile to multiple variants of %s target!\n", Target::ISAToString(ISA));
            return {Target::ISA::NUM_ISAS, -1};
        }

        // Check for incompatible SSE variants
        if ((ISA == Target::SSE42 && ISAIndices.find(Target::SSE41) != ISAIndices.end()) ||
            (ISA == Target::SSE41 && ISAIndices.find(Target::SSE42) != ISAIndices.end())) {
            Error(SourcePos(), "Can't compile to both SSE4.1 and SSE4.2 targets!\n");
            return {Target::ISA::NUM_ISAS, -1};
        }

        ISAIndices[ISA] = i;
    }

    // Find the first initialized target machine from the targets we
    // compiled to above.  We'll use this as the target machine for
    // compiling the dispatch module--this is safe in that it is the
    // least-common-denominator of all of the targets we compiled to.
    for (int i = 0; i < Target::NUM_ISAS; i++) {
        auto commonISA = static_cast<Target::ISA>(i);
        auto it = ISAIndices.find(commonISA);
        if (it != ISAIndices.end()) {
            return {commonISA, it->second};
        }
    }

    return {Target::ISA::NUM_ISAS, -1};
}

// Reset the target and module to nullptr.
// TODO!: ideally, we need to get rid of this global states.
void lResetTargetAndModule() {
    m = nullptr;
    g->target = nullptr;
}

// Reset the target and module to the given by index values in the given vectors.
void lResetTargetAndModule(std::vector<std::unique_ptr<Module>> &modules, std::vector<std::unique_ptr<Target>> &targets,
                           int i) {
    m = modules[i].get();
    g->target = targets[i].get();

    // It is important reinitialize the LLVM utils for each change of
    // target when Target constructor is not called.
    InitLLVMUtil(g->ctx, *g->target);
}

/**
 * Creates and outputs a dispatch module for multi-target compilation.
 *
 * When compiling for multiple target ISAs, this function generates a dispatch module
 * that contains dispatch functions for all exported functions. Each dispatch function
 * checks the CPU capabilities at runtime and calls the appropriate target-specific
 * implementation. The dispatch module also contains single definitions of all global
 * variables to avoid duplicate symbols across target-specific object files.
 */
int Module::GenerateDispatch(const char *srcFile, std::vector<ISPCTarget> &targets,
                             std::vector<std::unique_ptr<Module>> &modules,
                             std::vector<std::unique_ptr<Target>> &targetsPtrs, Output &output) {
    std::map<std::string, FunctionTargetVariants> exportedFunctions;

    // Also check if we have same ISA with different vector widths
    auto [commonISA, commonTargetIndex] = lCheckAndFillISAIndices(targets);
    if (commonTargetIndex == -1) {
        return 1;
    }

    ISPCTarget commonTarget = targets[commonTargetIndex];
    Assert(commonTarget != ISPCTarget::none);

    // Set the module that corresponds to the common target ISA
    lResetTargetAndModule(modules, targetsPtrs, commonTargetIndex);

    // Handle creating a "generic" header file for multiple targets
    // that use exported varyings
    DispatchHeaderInfo DHI;
    if (!DHI.initialize(output.header)) {
        return 1;
    }

    // Create the dispatch module,
    llvm::Module *dispatchModule = lInitDispatchModule();
    if (!dispatchModule) {
        Error(SourcePos(), "Failed to create dispatch module.\n");
        return 1;
    }

    // Fill in the dispatch module and write the header file for the dispatch.
    for (unsigned int i = 0; i < targets.size(); ++i) {
        lResetTargetAndModule(modules, targetsPtrs, i);

        if (output.flags.isDepsToStdout()) {
            // We need to fix that because earlier we set it to false to avoid
            // writing deps file with targets' suffixes.
            m->output.flags.setDepsToStdout(true);
        }

        // Extract globals unless already created; in the latter case, just
        // do the checking
        lExtractOrCheckGlobals(m->module, dispatchModule, i != 0);

        // Grab pointers to the exported functions from the module we
        // just compiled, for use in generating the dispatch function
        // later.
        lGetExportedFunctions(m->symbolTable, exportedFunctions);

        if (i == (targets.size() - 1)) {
            // only print backmatter on the last target.
            DHI.EmitBackMatter = true;
        }
        if (!output.header.empty() && !m->writeDispatchHeader(&DHI)) {
            return 1;
        }

        // Just precausiously reset observers to nullptr to avoid dangling pointers.
        lResetTargetAndModule();
    }

    // Set the module that corresponds to the common target ISA
    lResetTargetAndModule(modules, targetsPtrs, commonTargetIndex);

    // Create the dispatch functions and emit the dispatch module
    lEmitDispatchModule(dispatchModule, exportedFunctions);

    return WriteDispatchOutputFiles(dispatchModule, output);
}

/**
 * Prepares output configuration for a specific target in multi-target compilation.
 *
 * This function modifies the output configuration for each target, ensuring unique
 * filenames by appending the target's ISA name to output and header files. It also
 * handles special considerations for multi-target compilation:
 * - Mangles output and header filenames with target-specific suffixes
 * - Clears stub file names (not used in multi-target mode)
 * - Disables writing dependencies for individual targets
 */
static Module::Output lCreateTargetOutputs(Module::Output &output, ISPCTarget target) {
    Module::Output targetOutputs = output;

    // If a header or out filename was specified, mangle it with the target ISA name
    if (!targetOutputs.out.empty()) {
        targetOutputs.out = targetOutputs.OutFileNameTarget(g->target);
    }

    if (!targetOutputs.header.empty()) {
        targetOutputs.header = targetOutputs.HeaderFileNameTarget(g->target);
    }

    // TODO: --host-stub and --dev-stub are ignored in multi-target mode?
    // Clear host and device stub filenames, as they are not used in multi-target mode
    targetOutputs.hostStub = "";
    targetOutputs.devStub = "";

    // Disable writing dependencies file for individual targets.
    // deps file is written only once for all targets, so we will generate
    // it during the dispatch module generation.
    targetOutputs.deps = "";
    targetOutputs.flags.setDepsToStdout(false);

    return targetOutputs;
}

// Compiles the given source file for multiple target ISAs and creates a dispatch module.
int Module::CompileMultipleTargets(const char *srcFile, Arch arch, const char *cpu, std::vector<ISPCTarget> &targets,
                                   Output &output) {
    // The user supplied multiple targets
    Assert(targets.size() > 1);

    if (lValidateMultiTargetInputs(srcFile, output.out, cpu)) {
        return 1;
    }

    // Make sure that the function names for 'export'ed functions have
    // the target ISA appended to them.
    g->mangleFunctionsWithTarget = true;

    // This function manages multiple Module and Target instances, transferring
    // ownership to local vectors. These instances are automatically destroyed
    // when the function returns.
    std::vector<std::unique_ptr<Module>> modules;
    std::vector<std::unique_ptr<Target>> targetsPtrs;

    for (unsigned int i = 0; i < targets.size(); ++i) {
        auto targetPtr = Target::Create(arch, cpu, targets[i], output.flags.getPICLevel(), output.flags.getMCModel(),
                                        g->printTarget);
        if (!targetPtr) {
            return 1;
        }

        // Here, we transfer the ownership of the target to the vector, i.e.,
        // the lifetime of the target objects is tied to the function scope.
        // Same happens for the module objects.
        targetsPtrs.push_back(std::move(targetPtr));

        // Output and header names are set for each target with the target's suffix.
        Output targetOutputs = lCreateTargetOutputs(output, targets[i]);

        auto modulePtr = Module::Create(srcFile, targetOutputs, CompilationMode::Multiple);

        // Transfer the ownership of the module to the vector, i.e., the
        // lifetime of the module objects is tied to the function scope.
        modules.push_back(std::move(modulePtr));

        int compilerResult = m->CompileSingleTarget(arch, cpu, targets[i]);
        if (compilerResult) {
            return compilerResult;
        }

        // Important: Don't delete the llvm::Module *m here; we need to
        // keep it around so the llvm::Functions *s stay valid for when
        // we generate the dispatch module's functions...
        // Just precausiously reset observers to nullptr to avoid dangling pointers.
        lResetTargetAndModule();
    }

    // Generate the dispatch module
    return GenerateDispatch(srcFile, targets, modules, targetsPtrs, output);
}

int Module::CompileAndOutput(const char *srcFile, Arch arch, const char *cpu, std::vector<ISPCTarget> &targets,
                             Module::Output &output) {
    if (targets.size() == 0 || targets.size() == 1) {
        // We're only compiling to a single target
        ISPCTarget target = ISPCTarget::none;
        if (targets.size() == 1) {
            target = targets[0];
        }

        // Both the target and the module objects lifetime is tied to the scope of
        // this function. Raw pointers g->target and m are used as observers.
        // They are initialized inside Create functions.
        // Ideally, we should set m and g->target to nullptr after we are done,
        // i.e., after CompileSingleTarget returns, but we return from the
        // function immediately after that, so it is not necessary. Although,
        // one should be careful if something changes here in the future.
        auto targetPtr =
            Target::Create(arch, cpu, target, output.flags.getPICLevel(), output.flags.getMCModel(), g->printTarget);
        if (!targetPtr) {
            return 1;
        }
        auto modulePtr = Module::Create(srcFile, output);

        return m->CompileSingleTarget(arch, cpu, target);
    } else {
        return CompileMultipleTargets(srcFile, arch, cpu, targets, output);
    }
}

int Module::LinkAndOutput(std::vector<std::string> linkFiles, OutputType outputType, std::string outFileName) {
    auto llvmLink = std::make_unique<llvm::Module>("llvm-link", *g->ctx);
    llvm::Linker linker(*llvmLink);
    for (const auto &file : linkFiles) {
        llvm::SMDiagnostic err;
        std::unique_ptr<llvm::Module> m = nullptr;
        std::ifstream inputStream(file, std::ios::binary);
        if (!inputStream.is_open()) {
            perror(file.c_str());
            return 1;
        }
        if (lIsLlvmBitcode(inputStream)) {
            m = llvm::parseIRFile(file, err, *g->ctx);
        }
#ifdef ISPC_XE_ENABLED
        else if (lIsSpirVBitcode(inputStream)) {
            m = translateFromSPIRV(inputStream);
        }
#endif
        else {
            Error(SourcePos(), "Unrecognized format of input file %s", file.c_str());
            return 1;
        }
        if (m) {
            linker.linkInModule(std::move(m), 0);
        }
        inputStream.close();
    }
    if (!outFileName.empty()) {
        if ((outputType == Bitcode) || (outputType == BitcodeText)) {
            writeBitcode(llvmLink.get(), outFileName, outputType);
        }
#ifdef ISPC_XE_ENABLED
        else if (outputType == SPIRV) {
            writeSPIRV(llvmLink.get(), outFileName);
        }
#endif
        return 0;
    }
    return 1;
}
