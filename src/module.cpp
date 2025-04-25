/*
  Copyright (c) 2010-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file module.cpp
    @brief Impementation of the Module class, which collects the result of compiling
           a source file and then generates output (object files, etc.)
*/

#include "module.h"
#include "binary_type.h"
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
#include <ctype.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <utility>

#include <clang/Basic/CharInfo.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/FileSystemOptions.h>
#include <clang/Basic/MacroBuilder.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/Version.h>
#include <clang/Frontend/FrontendDiagnostic.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Frontend/PreprocessorOutputOptions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/Lex/HeaderSearch.h>
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Lex/ModuleLoader.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/PassRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Transforms/IPO.h>
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
// Note that this define must be after clang includes, as they undefining this symbol.
#define strcasecmp stricmp
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

/*! list of files encountered by the parser. this allows emitting of
    the module file's dependencies via the -MMM option */
std::set<std::string> registeredDependencies;

/* This set is used to store strings that is referenced in yylloc (SourcePos)
   in lexer once and not to lost memory via just strduping them. */
std::set<std::string> pseudoDependencies;

/*! this is where the parser tells us that it has seen the given file
    name in the CPP hash */
const char *RegisterDependency(const std::string &fileName) {
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

#if ISPC_LLVM_VERSION >= ISPC_LLVM_19_0
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
extern YY_BUFFER_STATE yy_scan_string(const char *);
extern YY_BUFFER_STATE yy_create_buffer(FILE *, int);
extern void yy_delete_buffer(YY_BUFFER_STATE);
extern void ParserInit();

int Module::preprocessAndParse() {
    llvm::SmallVector<llvm::StringRef, 6> refs;

    initCPPBuffer();

    const int numErrors = execPreprocessor(srcFile, bufferCPP->os.get(), g->preprocessorOutputType);
    errorCount += (g->ignoreCPPErrors) ? 0 : numErrors;

    if (g->onlyCPP) {
        return errorCount; // Return early
    }

    parseCPPBuffer();
    clearCPPBuffer();

    return 0;
}

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
        const ConstExpr *ce = llvm::dyn_cast<ConstExpr>(Optimize(expr));
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
                    initExpr = Optimize(initExpr);
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
    llvm::GlobalVariable *oldGV = nullptr;
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

        // Now, we either have a redeclaration of a global, or a definition
        // of a previously-declared global.  First, save the pointer to the
        // previous llvm::GlobalVariable
        oldGV = gv;
    } else {
        sym = new Symbol(name, pos, Symbol::SymbolKind::Variable, type, storageClass);
        symbolTable->AddVariable(sym);
    }
    sym->constValue = constValue;

    llvm::GlobalValue::LinkageTypes linkage =
        sym->storageClass.IsStatic() ? llvm::GlobalValue::InternalLinkage : llvm::GlobalValue::ExternalLinkage;

    // Note that the nullptr llvmInitializer is what leads to "extern"
    // declarations coming up extern and not defining storage (a bit
    // subtle)...
    sym->storageInfo = new AddressInfo(
        new llvm::GlobalVariable(*module, llvmType, isConst, linkage, llvmInitializer, sym->name.c_str()), llvmType);

    // Apply alignment if specified
    if (alignment > 0) {
        llvm::GlobalVariable *gv = llvm::cast<llvm::GlobalVariable>(sym->storageInfo->getPointer());
        gv->setAlignment(llvm::Align(alignment));
    }

    // Patch up any references to the previous GlobalVariable (e.g. from a
    // declaration of a global that was later defined.)
    if (oldGV != nullptr) {
        oldGV->replaceAllUsesWith(sym->storageInfo->getPointer());
        oldGV->removeFromParent();
        sym->storageInfo->getPointer()->setName(sym->name.c_str());
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
        /*#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6
                Assert(var.Verify());
        #else // LLVM 3.7+
              // comming soon
        #endif*/
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

static const std::vector<Module::OutputTypeInfo> outputTypeInfos = {
    /* Asm         */ {"assembly", {"s"}},
    /* Bitcode     */ {"LLVM bitcode", {"bc"}},
    /* BitcodeText */ {"LLVM assembly", {"ll"}},
    /* Object      */ {"object", {"o", "obj"}},
    /* Header      */ {"header", {"h", "hh", "hpp"}},
    /* Deps        */ {"dependencies", {}}, // No suffix
    /* DevStub     */ {"dev-side offload stub", {"c", "cc", "c++", "cxx", "cpp"}},
    /* HostStub    */ {"host-side offload stub", {"c", "cc", "c++", "cxx", "cpp"}},
    /* CPPStub     */ {"preprocessed stub", {"ispi", "i"}},
#ifdef ISPC_XE_ENABLED
    /* ZEBIN       */ {"L0 binary", {"bin"}},
    /* SPIRV       */ {"SPIR-V", {"spv"}},
#endif
    // Deps and other types that don't require warnings can be omitted
};

static void lReportInvalidSuffixWarning(std::string filename, Module::OutputType outputType) {
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

    lReportInvalidSuffixWarning(output.out, outputType);

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

// Translate ISPC CPU name to Neo representation
static std::string translateCPU(const std::string &CPU) {
    // For the platforms that we support it's 1:1 mapping at the moment,
    // in case of exceptions, they need to be handled here.
    return CPU;
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
    const std::string neoCPU = translateCPU(CPUName);

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
    oclocArgs.push_back(neoCPU.c_str());
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
                  "#endif\n"
                  "#ifndef __ISPC_ALIGNED_STRUCT__\n"
                  "#if defined(_MSC_VER)\n"
                  "// Visual Studio\n"
                  "#define __ISPC_ALIGNED_STRUCT__(s) __ISPC_ALIGN__(s) struct\n"
                  "#else\n"
                  "// Clang, GCC, ICC\n"
                  "#define __ISPC_ALIGNED_STRUCT__(s) struct __ISPC_ALIGN__(s)\n"
                  "#endif\n"
                  "#endif\n\n");

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

    lReportInvalidSuffixWarning(CO.deps, OutputType::Deps);

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

    lReportInvalidSuffixWarning(output.devStub, OutputType::DevStub);

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

bool Module::writeCPPStub() {
    // Get a file descriptor corresponding to where we want the output to
    // go.  If we open it, it'll be closed by the llvm::raw_fd_ostream
    // destructor.
    int fd = -1;
    int flags = O_CREAT | O_WRONLY | O_TRUNC;

    // TODO: open file using some LLVM API?
    if (output.out.empty()) {
        return false;
    } else if (output.out == "-") {
        fd = 1;
    } else {
#ifdef ISPC_HOST_IS_WINDOWS
        fd = _open(output.out.c_str(), flags, 0644);
#else
        fd = open(output.out.c_str(), flags, 0644);
#endif // ISPC_HOST_IS_WINDOWS
        if (fd == -1) {
            perror(output.out.c_str());
            return false;
        }
    }

    // The CPP stream should have been initialized: print and clean it up.
    Assert(bufferCPP && "`bufferCPP` should not be null");
    llvm::raw_fd_ostream fos(fd, (fd != 1), false);
    fos << bufferCPP->str;
    bufferCPP.reset();

    return true;
}

bool Module::writeHostStub() {
    FILE *file = fopen(output.hostStub.c_str(), "w");

    lReportInvalidSuffixWarning(output.hostStub, OutputType::HostStub);

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

bool Module::writeHeader() {
    FILE *f = fopen(output.header.c_str(), "w");

    lReportInvalidSuffixWarning(output.header, OutputType::Header);

    if (!f) {
        perror("fopen");
        return false;
    }
    fprintf(f, "//\n// %s\n// (Header automatically generated by the ispc compiler.)\n", output.header.c_str());
    fprintf(f, "// DO NOT EDIT THIS FILE.\n//\n\n");

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

    fclose(f);
    return true;
}

struct ispc::DispatchHeaderInfo {
    bool EmitUnifs = false;
    bool EmitFuncs = false;
    bool EmitFrontMatter = false;
    bool EmitBackMatter = false;
    bool Emit4 = false;
    bool Emit8 = false;
    bool Emit16 = false;
    FILE *file = nullptr;
    const char *fn = nullptr;
    std::string header{};

    bool initialize(std::string headerFileName) {
        EmitUnifs = true;
        EmitFuncs = true;
        EmitFrontMatter = true;
        // This is toggled later.
        EmitBackMatter = false;
        Emit4 = true;
        Emit8 = true;
        Emit16 = true;
        header = headerFileName;
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

    void closeFile() {
        if (file != nullptr) {
            fclose(file);
            file = nullptr;
        }
    }

    ~DispatchHeaderInfo() { closeFile(); }
};

bool Module::writeDispatchHeader(DispatchHeaderInfo *DHI) {
    FILE *f = DHI->file;

    lReportInvalidSuffixWarning(DHI->fn, OutputType::Header);

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

static void lInitializeSourceManager(clang::FrontendInputFile &input, clang::DiagnosticsEngine &diag,
                                     clang::FileManager &fileMgr, clang::SourceManager &srcMgr) {
    clang::SrcMgr::CharacteristicKind kind = clang::SrcMgr::C_User;
    if (input.isBuffer()) {
        srcMgr.setMainFileID(srcMgr.createFileID(input.getBuffer(), kind));
        Assert(srcMgr.getMainFileID().isValid() && "Couldn't establish MainFileID");
        return;
    }

    llvm::StringRef inputFile = input.getFile();

    // Figure out where to get and map in the main file.
    auto fileOrError = inputFile == "-" ? fileMgr.getSTDIN() : fileMgr.getFileRef(inputFile, true);
    if (!fileOrError) {
        // FIXME: include the error in the diagnostic even when it's not stdin.
        auto errCode = llvm::errorToErrorCode(fileOrError.takeError());
        if (inputFile != "-") {
            diag.Report(clang::diag::err_fe_error_reading) << inputFile;
        } else {
            diag.Report(clang::diag::err_fe_error_reading_stdin) << errCode.message();
        }
        return;
    }

    srcMgr.setMainFileID(srcMgr.createFileID(*fileOrError, clang::SourceLocation(), kind));
    Assert(srcMgr.getMainFileID().isValid() && "Couldn't establish MainFileID");
    return;
}

// Copied from InitPreprocessor.cpp
static bool lMacroBodyEndsInBackslash(llvm::StringRef MacroBody) {
    while (!MacroBody.empty() && clang::isWhitespace(MacroBody.back())) {
        MacroBody = MacroBody.drop_back();
    }
    return !MacroBody.empty() && MacroBody.back() == '\\';
}

// Copied from InitPreprocessor.cpp
// Append a #define line to Buf for Macro.  Macro should be of the form XXX,
// in which case we emit "#define XXX 1" or "XXX=Y z W" in which case we emit
// "#define XXX Y z W".  To get a #define with no value, use "XXX=".
static void lDefineBuiltinMacro(clang::MacroBuilder &Builder, llvm::StringRef Macro, clang::DiagnosticsEngine &Diags) {
    std::pair<llvm::StringRef, llvm::StringRef> MacroPair = Macro.split('=');
    llvm::StringRef MacroName = MacroPair.first;
    llvm::StringRef MacroBody = MacroPair.second;
    if (MacroName.size() != Macro.size()) {
        // Per GCC -D semantics, the macro ends at \n if it exists.
        llvm::StringRef::size_type End = MacroBody.find_first_of("\n\r");
        if (End != llvm::StringRef::npos) {
            Diags.Report(clang::diag::warn_fe_macro_contains_embedded_newline) << MacroName;
        }
        MacroBody = MacroBody.substr(0, End);
        // We handle macro bodies which end in a backslash by appending an extra
        // backslash+newline.  This makes sure we don't accidentally treat the
        // backslash as a line continuation marker.
        if (lMacroBodyEndsInBackslash(MacroBody)) {
            Builder.defineMacro(MacroName, llvm::Twine(MacroBody) + "\\\n");
        } else {
            Builder.defineMacro(MacroName, MacroBody);
        }
    } else {
        // Push "macroname 1".
        Builder.defineMacro(Macro);
    }
}

void lCreateVirtualHeader(clang::FileManager &fileMgr, clang::SourceManager &srcMgr, llvm::StringRef Filename,
                          llvm::StringRef Content) {
    std::unique_ptr<llvm::MemoryBuffer> Buffer = llvm::MemoryBuffer::getMemBuffer(Content);
    clang::FileEntryRef FE = fileMgr.getVirtualFileRef(Filename, Buffer->getBufferSize(), 0);
    srcMgr.overrideFileContents(FE, std::move(Buffer));
}

void lBuilderAppendHeader(clang::MacroBuilder &Builder, std::string &header) {
    Builder.append(llvm::Twine("#include \"") + header + "\"");
}

using StringRefFunc = llvm::StringRef (*)();
void lAddImplicitInclude(clang::MacroBuilder &Builder, clang::FileManager &fileMgr, clang::SourceManager &srcMgr,
                         const char *header, StringRefFunc contentFunc) {
#ifdef ISPC_HOST_IS_WINDOWS
    std::string headerPath = "c:/";
#else
    std::string headerPath = "/";
#endif // ISPC_HOST_IS_WINDOWS

    std::string name = header;
    // Slim binary fetches headers from the file system whereas composite
    // binary fetches them from in-memory buffers creating virtual files.
    if (g->isSlimBinary) {
        lBuilderAppendHeader(Builder, name);
    } else {
        name = headerPath + name;
        lBuilderAppendHeader(Builder, name);
        lCreateVirtualHeader(fileMgr, srcMgr, name, contentFunc());
    }
}

// Core logic of buiding macros copied from clang::InitializePreprocessor from InitPreprocessor.cpp
static void lInitializePreprocessor(clang::Preprocessor &PP, clang::PreprocessorOptions &InitOpts,
                                    clang::FileManager &fileMgr, clang::SourceManager &srcMgr) {
    std::string PredefineBuffer;
    PredefineBuffer.reserve(4080);
    llvm::raw_string_ostream Predefines(PredefineBuffer);
    clang::MacroBuilder Builder(Predefines);

    // Process #define's and #undef's in the order they are given.
    for (unsigned i = 0, e = InitOpts.Macros.size(); i != e; ++i) {
        if (InitOpts.Macros[i].second) {
            Builder.undefineMacro(InitOpts.Macros[i].first);
        } else {
            lDefineBuiltinMacro(Builder, InitOpts.Macros[i].first, PP.getDiagnostics());
        }
    }

    if (!g->genStdlib) {
        lAddImplicitInclude(Builder, fileMgr, srcMgr, "core.isph", getCoreISPHRef);
        if (g->includeStdlib) {
            lAddImplicitInclude(Builder, fileMgr, srcMgr, "stdlib.isph", getStdlibISPHRef);
        }
    }

    // Copy PredefinedBuffer into the Preprocessor.
    PP.setPredefines(std::move(PredefineBuffer));
}

// Initilialize PreprocessorOutputOptions with ISPC PreprocessorOutputType
// Return if the output can be treated as preprocessor output
static bool lSetPreprocessorOutputOptions(clang::PreprocessorOutputOptions *opts,
                                          Globals::PreprocessorOutputType preprocessorOutputType) {
    switch (preprocessorOutputType) {
    case Globals::PreprocessorOutputType::Cpp:
        // Don't remove comments in the preprocessor, so that we can accurately
        // track the source file position by handling them ourselves.
        opts->ShowComments = 1;
        opts->ShowCPP = 1;
        return true;
    case Globals::PreprocessorOutputType::WithMacros:
        opts->ShowCPP = 1;
        opts->ShowMacros = 1;
        opts->ShowComments = 1;
        return true;
    case Globals::PreprocessorOutputType::MacrosOnly:
        opts->ShowMacros = 1;
        return false;
    }

    // Undefined PreprocessorOutputType
    return false;
}

static void lSetHeaderSeachOptions(const std::shared_ptr<clang::HeaderSearchOptions> opts) {
    opts->UseBuiltinIncludes = 0;
    opts->UseStandardSystemIncludes = 0;
    opts->UseStandardCXXIncludes = 0;
    if (g->debugPrint) {
        opts->Verbose = 1;
    }
    for (int i = 0; i < (int)g->includePath.size(); ++i) {
        opts->AddPath(g->includePath[i], clang::frontend::Angled, false /* not a framework */,
                      true /* ignore sys root */);
    }
}

static void lSetTargetSpecificMacroDefinitions(const std::shared_ptr<clang::PreprocessorOptions> opts) {
    // Add #define for current compilation target
    char targetMacro[128];
    snprintf(targetMacro, sizeof(targetMacro), "ISPC_TARGET_%s", g->target->GetISAString());
    char *p = targetMacro;
    while (*p) {
        *p = toupper(*p);
        if ((*p == '-') || (*p == '.')) {
            *p = '_';
        }
        ++p;
    }
    opts->addMacroDef(targetMacro);

    // Add 'TARGET_WIDTH' macro to expose vector width to user.
    std::string target_width = "TARGET_WIDTH=" + std::to_string(g->target->getVectorWidth());
    opts->addMacroDef(target_width);

    // Add 'TARGET_ELEMENT_WIDTH' macro to expose element width to user.
    std::string target_element_width = "TARGET_ELEMENT_WIDTH=" + std::to_string(g->target->getDataTypeWidth() / 8);
    opts->addMacroDef(target_element_width);

    if (g->target->hasHalfConverts()) {
        opts->addMacroDef("ISPC_TARGET_HAS_HALF");
    }
    if (g->target->hasHalfFullSupport()) {
        opts->addMacroDef("ISPC_TARGET_HAS_HALF_FULL_SUPPORT");
    }
    if (g->target->hasRand()) {
        opts->addMacroDef("ISPC_TARGET_HAS_RAND");
    }
    if (g->target->hasTranscendentals()) {
        opts->addMacroDef("ISPC_TARGET_HAS_TRANSCENDENTALS");
    }
    if (g->target->hasTrigonometry()) {
        opts->addMacroDef("ISPC_TARGET_HAS_TRIGONOMETRY");
    }
    if (g->target->hasRsqrtd()) {
        opts->addMacroDef("ISPC_TARGET_HAS_RSQRTD");
    }
    if (g->target->hasRcpd()) {
        opts->addMacroDef("ISPC_TARGET_HAS_RCPD");
    }
    if (g->target->hasSatArith()) {
        opts->addMacroDef("ISPC_TARGET_HAS_SATURATING_ARITHMETIC");
    }
    if (g->target->hasIntelVNNI()) {
        opts->addMacroDef("ISPC_TARGET_HAS_INTEL_VNNI");
    }
    if (g->target->hasIntelVNNI_Int8()) {
        opts->addMacroDef("ISPC_TARGET_HAS_INTEL_VNNI_INT8");
    }
    if (g->target->hasIntelVNNI_Int16()) {
        opts->addMacroDef("ISPC_TARGET_HAS_INTEL_VNNI_INT16");
    }
    if (g->target->hasConflictDetection()) {
        opts->addMacroDef("ISPC_TARGET_HAS_CONFLICT_DETECTION");
    }
    // TODO! what is the problem to have g->target->hasXePrefetch function returning bool for non XE_ENABLED builds??
#ifdef ISPC_XE_ENABLED
    if (g->target->hasXePrefetch()) {
        opts->addMacroDef("ISPC_TARGET_HAS_XE_PREFETCH");
    }
#endif

    // Define mask bits
    std::string ispc_mask_bits = "ISPC_MASK_BITS=" + std::to_string(g->target->getMaskBitCount());
    opts->addMacroDef(ispc_mask_bits);

    if (g->target->is32Bit()) {
        opts->addMacroDef("ISPC_POINTER_SIZE=32");
    } else {
        opts->addMacroDef("ISPC_POINTER_SIZE=64");
    }

    if (g->target->hasFp16Support()) {
        // TODO! rename/alias to ISPC_TARGET_HAS_FP16_SUPPORT
        opts->addMacroDef("ISPC_FP16_SUPPORTED");
    }

    if (g->target->hasFp64Support()) {
        // TODO! rename/alias to ISPC_TARGET_HAS_FP64_SUPPORT
        opts->addMacroDef("ISPC_FP64_SUPPORTED");
    }
}

static void lSetCmdlineDependentMacroDefinitions(const std::shared_ptr<clang::PreprocessorOptions> opts) {
    if (g->opt.disableAsserts) {
        opts->addMacroDef("ISPC_ASSERTS_DISABLED");
    }

    if (g->opt.forceAlignedMemory) {
        opts->addMacroDef("ISPC_FORCE_ALIGNED_MEMORY");
    }

    if (g->opt.fastMaskedVload) {
        opts->addMacroDef("ISPC_FAST_MASKED_VLOAD");
    }

    if (g->enableLLVMIntrinsics) {
        opts->addMacroDef("ISPC_LLVM_INTRINSICS_ENABLED");
    }

    std::string math_lib = "ISPC_MATH_LIB_VAL=" + std::to_string((int)g->mathLib);
    opts->addMacroDef(math_lib);

    std::string memory_alignment = "ISPC_MEMORY_ALIGNMENT_VAL=";
    if (g->forceAlignment != -1) {
        memory_alignment += std::to_string(g->forceAlignment);
    } else {
        auto width = g->target->getVectorWidth();
        if (width == 1) {
            // Do we have a scalar target?
            // It was in m4 like this:
            // ;; ifelse(WIDTH, 1, `define(`ALIGNMENT', `16')', `define(`ALIGNMENT', `eval(WIDTH*4)')')
            memory_alignment += std::to_string(16);
        } else {
            memory_alignment += std::to_string(width * 4);
        }
    }
    opts->addMacroDef(memory_alignment);

    if (g->target->hasArmDotProduct()) {
        opts->addMacroDef("ISPC_TARGET_HAS_ARM_DOT_PRODUCT");
    }
    if (g->target->hasArmI8MM()) {
        opts->addMacroDef("ISPC_TARGET_HAS_ARM_I8MM");
    }
}

static void lSetPreprocessorOptions(const std::shared_ptr<clang::PreprocessorOptions> opts) {
    if (g->includeStdlib) {
        opts->addMacroDef("ISPC_INCLUDE_STDLIB");
    }

    std::string math_lib_ispc = "ISPC_MATH_LIB_ISPC_VAL=" + std::to_string((int)Globals::MathLib::Math_ISPC);
    opts->addMacroDef(math_lib_ispc);
    std::string math_lib_ispc_fast =
        "ISPC_MATH_LIB_ISPC_FAST_VAL=" + std::to_string((int)Globals::MathLib::Math_ISPCFast);
    opts->addMacroDef(math_lib_ispc_fast);
    std::string math_lib_svml = "ISPC_MATH_LIB_SVML_VAL=" + std::to_string((int)Globals::MathLib::Math_SVML);
    opts->addMacroDef(math_lib_svml);
    std::string math_lib_system = "ISPC_MATH_LIB_SYSTEM_VAL=" + std::to_string((int)Globals::MathLib::Math_System);
    opts->addMacroDef(math_lib_system);

    constexpr int buf_size = 25;
    char ispc_major[buf_size], ispc_minor[buf_size];
    snprintf(ispc_major, buf_size, "ISPC_MAJOR_VERSION=%d", ISPC_VERSION_MAJOR);
    snprintf(ispc_minor, buf_size, "ISPC_MINOR_VERSION=%d", ISPC_VERSION_MINOR);
    opts->addMacroDef(ispc_major);
    opts->addMacroDef(ispc_minor);

    char llvm_major[buf_size], llvm_minor[buf_size];
    snprintf(llvm_major, buf_size, "LLVM_VERSION_MAJOR=%d", LLVM_VERSION_MAJOR);
    snprintf(llvm_minor, buf_size, "LLVM_VERSION_MINOR=%d", LLVM_VERSION_MINOR);
    opts->addMacroDef(llvm_major);
    opts->addMacroDef(llvm_minor);

    // Target specific macro definitions
    lSetTargetSpecificMacroDefinitions(opts);

    // Set macro definitions that depends on command line flags of ISPC invocation.
    lSetCmdlineDependentMacroDefinitions(opts);

    for (unsigned int i = 0; i < g->cppArgs.size(); ++i) {
        // Sanity check--should really begin with -D
        if (g->cppArgs[i].substr(0, 2) == "-D") {
            opts->addMacroDef(g->cppArgs[i].substr(2));
        }
    }
}

static void lSetLangOptions(clang::LangOptions *opts) { opts->LineComment = 1; }

int Module::execPreprocessor(const char *infilename, llvm::raw_string_ostream *ostream,
                             Globals::PreprocessorOutputType preprocessorOutputType) const {
    clang::FrontendInputFile inputFile(infilename, clang::InputKind());
    llvm::raw_fd_ostream stderrRaw(2, false);

    // Create Diagnostic engine
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagIDs(new clang::DiagnosticIDs);
    llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOptions(new clang::DiagnosticOptions);
    clang::TextDiagnosticPrinter diagPrinter(stderrRaw, diagOptions.get(), false);
    clang::DiagnosticsEngine diagEng(diagIDs, diagOptions, &diagPrinter, false);

    diagEng.setSuppressAllDiagnostics(g->ignoreCPPErrors);

    // Create TargetInfo
    const std::shared_ptr<clang::TargetOptions> tgtOpts = std::make_shared<clang::TargetOptions>();
    llvm::Triple triple(module->getTargetTriple());
    if (triple.getTriple().empty()) {
        triple.setTriple(llvm::sys::getDefaultTargetTriple());
    }
    tgtOpts->Triple = triple.getTriple();
    clang::TargetInfo *tgtInfo = clang::TargetInfo::CreateTargetInfo(diagEng, tgtOpts);

    // Create and initialize PreprocessorOutputOptions
    clang::PreprocessorOutputOptions preProcOutOpts;
    bool isPreprocessedOutput = lSetPreprocessorOutputOptions(&preProcOutOpts, preprocessorOutputType);

    // Create and initialize HeaderSearchOptions
    const std::shared_ptr<clang::HeaderSearchOptions> hdrSearchOpts = std::make_shared<clang::HeaderSearchOptions>();
    lSetHeaderSeachOptions(hdrSearchOpts);

    // Create and initialize PreprocessorOptions
    const std::shared_ptr<clang::PreprocessorOptions> preProcOpts = std::make_shared<clang::PreprocessorOptions>();
    lSetPreprocessorOptions(preProcOpts);

    // Create and initialize LangOptions
    clang::LangOptions langOpts;
    lSetLangOptions(&langOpts);

    // Create and initialize SourceManager
    clang::FileSystemOptions fsOpts;
    clang::FileManager fileMgr(fsOpts);
    clang::SourceManager srcMgr(diagEng, fileMgr);
    lInitializeSourceManager(inputFile, diagEng, fileMgr, srcMgr);

// Create HeaderSearch and apply HeaderSearchOptions
#if ISPC_LLVM_VERSION >= ISPC_LLVM_21_0
    clang::HeaderSearch hdrSearch(*hdrSearchOpts, srcMgr, diagEng, langOpts, tgtInfo);
#else
    clang::HeaderSearch hdrSearch(hdrSearchOpts, srcMgr, diagEng, langOpts, tgtInfo);
#endif
    clang::ApplyHeaderSearchOptions(hdrSearch, *hdrSearchOpts, langOpts, triple);

    // Finally, create an preprocessor object
    clang::TrivialModuleLoader modLoader;
#if ISPC_LLVM_VERSION >= ISPC_LLVM_21_0
    clang::Preprocessor prep(*preProcOpts, diagEng, langOpts, srcMgr, hdrSearch, modLoader);
#else
    clang::Preprocessor prep(preProcOpts, diagEng, langOpts, srcMgr, hdrSearch, modLoader);
#endif

    // Initialize preprocessor
    prep.Initialize(*tgtInfo);
    prep.setPreprocessedOutput(isPreprocessedOutput);
    lInitializePreprocessor(prep, *preProcOpts, fileMgr, srcMgr);

    // do actual preprocessing
    diagPrinter.BeginSourceFile(langOpts, &prep);
    clang::DoPrintPreprocessedInput(prep, ostream, preProcOutOpts);
    diagPrinter.EndSourceFile();

    // deallocate some objects
    diagIDs.reset();
    diagOptions.reset();
    delete tgtInfo;

    // Return preprocessor diagnostic errors after processing
    return static_cast<int>(diagEng.hasErrorOccurred());
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

#if ISPC_LLVM_VERSION >= ISPC_LLVM_19_0
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

void Module::initCPPBuffer() {
    // If the CPP stream has been initialized, we have unexpected behavior.
    if (bufferCPP) {
        Assert("CPP stream has already been initialized.");
    }

    // Replace the CPP stream with a newly allocated one.
    bufferCPP.reset(new CPPBuffer{});
}

void Module::parseCPPBuffer() {
    YY_BUFFER_STATE strbuf = yy_scan_string(bufferCPP->str.c_str());
    yyparse();
    yy_delete_buffer(strbuf);
}

void Module::clearCPPBuffer() {
    if (bufferCPP) {
        bufferCPP.reset();
    }
}
