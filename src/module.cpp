/*
  Copyright (c) 2010-2024, Intel Corporation

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
#include <ctype.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

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
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#if ISPC_LLVM_VERSION >= ISPC_LLVM_17_0
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#ifdef ISPC_XE_ENABLED
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#endif
#include <llvm/Target/TargetIntrinsicInfo.h>

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

/*! list of files encountered by the parser. this allows emitting of
    the module file's dependencies via the -MMM option */
std::set<std::string> registeredDependencies;

/* This set is used to store strings that is referenced in yylloc (SourcePos)
   in lexer once and not to lost memory via just strduping them. */
std::set<std::string> pseudoDependencies;

/*! this is where the parser tells us that it has seen the given file
    name in the CPP hash */
const char *RegisterDependency(const std::string &fileName) {
    if (fileName[0] != '<' && fileName != "stdlib.ispc") {
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

Module::Module(const char *fn) : filename(fn) {
    // It's a hack to do this here, but it must be done after the target
    // information has been set (so e.g. the vector width is known...)  In
    // particular, if we're compiling to multiple targets with different
    // vector widths, this needs to be redone each time through.
    InitLLVMUtil(g->ctx, *g->target);

    symbolTable = new SymbolTable;
    ast = new AST;

    lDeclareSizeAndPtrIntTypes(symbolTable);

    module = new llvm::Module(!IsStdin(filename) ? filename : "<stdin>", *g->ctx);
    module->setTargetTriple(g->target->GetTripleString());

    // DataLayout information supposed to be managed in single place in Target class.
    module->setDataLayout(g->target->getDataLayout()->getStringRepresentation());
    lSetCodeModel(module);
    lSetPICLevel(module);

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
        if (IsStdin(filename)) {
            // Unfortunately we can't yet call Error() since the global 'm'
            // variable hasn't been initialized yet.
            Error(SourcePos(), "Can't emit debugging information with no source file on disk.\n");
            ++errorCount;
            delete diBuilder;
            diBuilder = nullptr;
        } else {
            auto [directory, name] = GetDirectoryAndFileName(g->currentDirectory, filename);
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

Module::~Module() {
    if (symbolTable)
        delete symbolTable;
    if (ast)
        delete ast;
    if (module)
        delete module;
    if (diBuilder)
        delete diBuilder;
}

extern FILE *yyin;
extern int yyparse();
typedef struct yy_buffer_state *YY_BUFFER_STATE;
extern void yy_switch_to_buffer(YY_BUFFER_STATE);
extern YY_BUFFER_STATE yy_scan_string(const char *);
extern YY_BUFFER_STATE yy_create_buffer(FILE *, int);
extern void yy_delete_buffer(YY_BUFFER_STATE);
extern void ParserInit();

int Module::CompileFile() {
    llvm::TimeTraceScope CompileFileTimeScope(
        "CompileFile", llvm::StringRef(filename + ("_" + std::string(g->target->GetISAString()))));
    ParserInit();

    // FIXME: it'd be nice to do this in the Module constructor, but this
    // function ends up calling into routines that expect the global
    // variable 'm' to be initialized and available (which it isn't until
    // the Module constructor returns...)
    {
        llvm::TimeTraceScope TimeScope("DefineStdlib");
        DefineStdlib(symbolTable, g->ctx, module, g->includeStdlib);
    }

    bool runPreprocessor = g->runCPP;

    if (runPreprocessor) {
        llvm::TimeTraceScope TimeScope("Frontend parser");
        if (!IsStdin(filename)) {
            // Try to open the file first, since otherwise we crash in the
            // preprocessor if the file doesn't exist.
            FILE *f = fopen(filename, "r");
            if (!f) {
                perror(filename);
                return 1;
            }
            fclose(f);
        }

        // If the CPP stream has been initialized, we have unexpected behavior.
        if (bufferCPP) {
            perror(filename);
            return 1;
        }

        // Replace the CPP stream with a newly allocated one.
        bufferCPP.reset(new CPPBuffer{});

        const int numErrors = execPreprocessor(!IsStdin(filename) ? filename : "-", bufferCPP->os.get());
        errorCount += (g->ignoreCPPErrors) ? 0 : numErrors;

        if (g->onlyCPP) {
            return errorCount; // Return early
        }

        YY_BUFFER_STATE strbuf = yy_scan_string(bufferCPP->str.c_str());
        yyparse();
        yy_delete_buffer(strbuf);
        clearCPPBuffer();
    } else {
        llvm::TimeTraceScope TimeScope("Frontend parser");
        // No preprocessor, just open up the file if it's not stdin..
        FILE *f = nullptr;
        if (IsStdin(filename)) {
            f = stdin;
        } else {
            f = fopen(filename, "r");
            if (f == nullptr) {
                perror(filename);
                return 1;
            }
        }
        yyin = f;
        yy_switch_to_buffer(yy_create_buffer(yyin, 4096));
        yyparse();
        fclose(f);
    }

    ast->Print(g->astDump);

    if (g->NoOmitFramePointer)
        for (llvm::Function &f : *module)
            f.addFnAttr("frame-pointer", "all");
    for (llvm::Function &f : *module)
        g->target->markFuncWithTargetAttr(&f);
    ast->GenerateIR();

    if (diBuilder)
        diBuilder->finalize();
    llvm::TimeTraceScope TimeScope("Optimize");
    if (errorCount == 0)
        Optimize(module, g->opt.level);

    return errorCount;
}

Symbol *Module::AddLLVMIntrinsicDecl(const std::string &name, ExprList *args, SourcePos pos) {
    if (g->enableLLVMIntrinsics == false) {
        Error(SourcePos(), "Calling LLVM intrinsics from ISPC source code is an experimental feature,"
                           " which can be enabled by passing \"--enable-llvm-intrinsics\" switch to the compiler.\n");
        return nullptr;
    }

    llvm::Function *funcDecl = nullptr;
#ifdef ISPC_XE_ENABLED
    if (g->target->isXeTarget()) {
        llvm::GenXIntrinsic::ID ID = llvm::GenXIntrinsic::lookupGenXIntrinsicID(name);
        if (ID == llvm::GenXIntrinsic::not_any_intrinsic) {
            Error(pos, "LLVM intrinsic \"%s\" not supported.", name.c_str());
            return nullptr;
        }
        std::vector<llvm::Type *> exprType;
        int nInits = args->exprs.size();
        if (llvm::GenXIntrinsic::isOverloadedRet(ID) || llvm::GenXIntrinsic::isOverloadedArg(ID, nInits)) {
            for (int i = 0; i < nInits; ++i) {
                const Type *argType = (args->exprs[i])->GetType();
                Assert(argType);
                exprType.push_back(argType->LLVMType(g->ctx));
            }
        }
        llvm::ArrayRef<llvm::Type *> argArr(exprType);
        funcDecl = llvm::GenXIntrinsic::getGenXDeclaration(module, ID, argArr);
        if (funcDecl) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
            // ReadNone, ReadOnly and WriteOnly are not supported for intrinsics anymore:
            FixFunctionAttribute(*funcDecl, llvm::Attribute::ReadNone, llvm::MemoryEffects::none());
            FixFunctionAttribute(*funcDecl, llvm::Attribute::ReadOnly, llvm::MemoryEffects::readOnly());
            FixFunctionAttribute(*funcDecl, llvm::Attribute::WriteOnly, llvm::MemoryEffects::writeOnly());
#endif
        }
    }
#endif
    if (!g->target->isXeTarget()) {
        llvm::TargetMachine *targetMachine = g->target->GetTargetMachine();
        const llvm::TargetIntrinsicInfo *TII = targetMachine->getIntrinsicInfo();
        llvm::Intrinsic::ID ID = llvm::Function::lookupIntrinsicID(llvm::StringRef(name));
        if (ID == llvm::Intrinsic::not_intrinsic && TII) {
            ID = static_cast<llvm::Intrinsic::ID>(TII->lookupName(llvm::StringRef(name)));
        }
        if (ID == llvm::Intrinsic::not_intrinsic) {
            Error(pos, "LLVM intrinsic \"%s\" not supported.", name.c_str());
            return nullptr;
        }
        std::vector<llvm::Type *> exprType;
        if (llvm::Intrinsic::isOverloaded(ID)) {
            int nInits = args->exprs.size();
            for (int i = 0; i < nInits; ++i) {
                const Type *argType = (args->exprs[i])->GetType();
                Assert(argType);
                exprType.push_back(argType->LLVMType(g->ctx));
            }
        }
        llvm::ArrayRef<llvm::Type *> argArr(exprType);
        funcDecl = llvm::Intrinsic::getDeclaration(module, ID, argArr);
        llvm::StringRef funcName = funcDecl->getName();

        if (g->target->checkIntrinsticSupport(funcName, pos) == false) {
            return nullptr;
        }
    }

    Assert(funcDecl != nullptr);
    Symbol *funcSym = CreateISPCSymbolForLLVMIntrinsic(funcDecl, symbolTable);
    return funcSym;
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

void Module::AddGlobalVariable(const std::string &name, const Type *type, Expr *initExpr, bool isConst,
                               StorageClass storageClass, SourcePos pos) {
    // These may be nullptr due to errors in parsing; just gracefully return
    // here if so.
    if (name == "" || type == nullptr) {
        Assert(errorCount > 0);
        return;
    }

    if (symbolTable->LookupFunction(name.c_str())) {
        Error(pos, "Global variable \"%s\" shadows previously-declared function.", name.c_str());
        return;
    }

    if (symbolTable->LookupFunctionTemplate(name.c_str())) {
        Error(pos, "Global variable \"%s\" shadows previously-declared function template.", name.c_str());
        return;
    }

    if (storageClass == SC_EXTERN_C) {
        Error(pos, "extern \"C\" qualifier can only be used for functions.");
        return;
    }

    if (storageClass == SC_EXTERN_SYCL) {
        Error(pos, "extern \"SYCL\" qualifier can only be used for functions.");
        return;
    }

    if (type->IsVoidType()) {
        Error(pos, "\"void\" type global variable is illegal.");
        return;
    }

    type = ArrayType::SizeUnsizedArrays(type, initExpr);
    if (type == nullptr)
        return;

    const ArrayType *at = CastType<ArrayType>(type);
    if (at != nullptr && at->TotalElementCount() == 0) {
        Error(pos, "Illegal to declare a global variable with unsized "
                   "array dimensions that aren't set with an initializer "
                   "expression.");
        return;
    }

    llvm::Type *llvmType = type->LLVMStorageType(g->ctx);
    if (llvmType == nullptr)
        return;

    // See if we have an initializer expression for the global.  If so,
    // make sure it's a compile-time constant!
    llvm::Constant *llvmInitializer = nullptr;
    ConstExpr *constValue = nullptr;
    if (storageClass == SC_EXTERN) {
        if (initExpr != nullptr)
            Error(pos, "Initializer can't be provided with \"extern\" global variable \"%s\".", name.c_str());
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
                        if ((storageClass != SC_STATIC) && (initPair.second == true)) {
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

                        if (type->IsConstType())
                            // Try to get a ConstExpr associated with
                            // the symbol.  This llvm::dyn_cast can
                            // validly fail, for example for types like
                            // StructTypes where a ConstExpr can't
                            // represent their values.
                            constValue = llvm::dyn_cast<ConstExpr>(initExpr);
                    } else
                        Error(initExpr->pos, "Initializer for global variable \"%s\" must be a constant.",
                              name.c_str());
                }
            }
        }

        // If no initializer was provided or if we couldn't get a value
        // above, initialize it with zeros..
        if (llvmInitializer == nullptr)
            llvmInitializer = llvm::Constant::getNullValue(llvmType);
    }

    Symbol *sym = symbolTable->LookupVariable(name.c_str());
    llvm::GlobalVariable *oldGV = nullptr;
    if (sym != nullptr) {
        // We've already seen either a declaration or a definition of this
        // global.

        // If the type doesn't match with the previous one, issue an error.
        if (!Type::Equal(sym->type, type) ||
            (sym->storageClass != SC_EXTERN && sym->storageClass != SC_EXTERN_C &&
             sym->storageClass != SC_EXTERN_SYCL && sym->storageClass != storageClass)) {
            Error(pos, "Definition of variable \"%s\" conflicts with definition at %s:%d.", name.c_str(), sym->pos.name,
                  sym->pos.first_line);
            return;
        }

        llvm::GlobalVariable *gv = llvm::dyn_cast<llvm::GlobalVariable>(sym->storageInfo->getPointer());
        Assert(gv != nullptr);

        // And issue an error if this is a redefinition of a variable
        if (gv->hasInitializer() && sym->storageClass != SC_EXTERN && sym->storageClass != SC_EXTERN_C &&
            sym->storageClass != SC_EXTERN_SYCL) {
            Error(pos, "Redefinition of variable \"%s\" is illegal. (Previous definition at %s:%d.)", sym->name.c_str(),
                  sym->pos.name, sym->pos.first_line);
            return;
        }

        // Now, we either have a redeclaration of a global, or a definition
        // of a previously-declared global.  First, save the pointer to the
        // previous llvm::GlobalVariable
        oldGV = gv;
    } else {
        sym = new Symbol(name, pos, type, storageClass);
        symbolTable->AddVariable(sym);
    }
    sym->constValue = constValue;

    llvm::GlobalValue::LinkageTypes linkage =
        (sym->storageClass == SC_STATIC) ? llvm::GlobalValue::InternalLinkage : llvm::GlobalValue::ExternalLinkage;

    // Note that the nullptr llvmInitializer is what leads to "extern"
    // declarations coming up extern and not defining storage (a bit
    // subtle)...
    sym->storageInfo = new AddressInfo(
        new llvm::GlobalVariable(*module, llvmType, isConst, linkage, llvmInitializer, sym->name.c_str()), llvmType);

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
            diSpace, name, name, file, pos.first_line, sym->type->GetDIType(diSpace), (sym->storageClass == SC_STATIC));
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
        for (int i = 0; i < st->GetElementCount(); ++i)
            if (!lRecursiveCheckValidParamType(st->GetElementType(i), soaOk, vectorOk, name, pos))
                return false;
        return true;
    }

    // Vector types are also not supported, pending ispc properly
    // supporting the platform ABI.  (Pointers to vector types are ok,
    // though.)  (https://github.com/ispc/ispc/issues/363)...
    if (vectorOk == false && CastType<VectorType>(t) != nullptr)
        return false;

    const SequentialType *seqt = CastType<SequentialType>(t);
    if (seqt != nullptr)
        return lRecursiveCheckValidParamType(seqt->GetElementType(), soaOk, vectorOk, name, pos);

    const PointerType *pt = CastType<PointerType>(t);
    if (pt != nullptr) {
        // Only allow exported uniform pointers
        // Uniform pointers to varying data, however, are ok.
        if (pt->IsVaryingType())
            return false;
        else
            return lRecursiveCheckValidParamType(pt->GetBaseType(), true, true, name, pos);
    }

    if (t->IsSOAType() && soaOk) {
        Warning(pos, "Exported function parameter \"%s\" points to SOA type.", name.c_str());
        return false;
    }

    if (t->IsVaryingType() && !vectorOk)
        return false;
    else {
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
        if (CastType<StructType>(type->GetBaseType()))
            Error(pos, "Struct parameter \"%s\" with vector typed member(s) is illegal in an exported function.",
                  name.c_str());
        else if (CastType<VectorType>(type))
            Error(pos, "Vector-typed parameter \"%s\" is illegal in an exported function.", name.c_str());
        else
            Error(pos, "Varying parameter \"%s\" is illegal in an exported function.", name.c_str());
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
                                    StorageClass storageClass, bool isInline, bool isNoInline, bool isVectorCall,
                                    bool isRegCall, SourcePos pos) {
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
            if (Type::Equal(overloadFunc->type, functionType))
                return;

            if (functionType->isExported || overloadType->isExported)
                Error(pos,
                      "Illegal to provide \"export\" qualifier for "
                      "functions with the same name but different types. "
                      "(Previous function declaration (%s:%d).)",
                      overloadFunc->pos.name, overloadFunc->pos.first_line);

            // If all of the parameter types match but the return type is
            // different, return an error--overloading by return type isn't
            // allowed.
            const FunctionType *ofType = CastType<FunctionType>(overloadFunc->type);
            Assert(ofType != nullptr);
            if (ofType->GetNumParameters() == functionType->GetNumParameters()) {
                int i;
                for (i = 0; i < functionType->GetNumParameters(); ++i) {
                    if (Type::Equal(ofType->GetParameterType(i), functionType->GetParameterType(i)) == false)
                        break;
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

    if (storageClass == SC_EXTERN_C || storageClass == SC_EXTERN_SYCL) {
        // Make sure the user hasn't supplied both an 'extern "C"' and a
        // 'task' qualifier with the function
        if (functionType->isTask) {
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
            if (Type::Equal(funcs[0]->type, functionType))
                return;
            else {
                Error(pos, "Can't overload extern \"C\" function \"%s\".", name.c_str());
                return;
            }
        }
    }

    // Get the LLVM FunctionType
    bool disableMask = (storageClass == SC_EXTERN_C || storageClass == SC_EXTERN_SYCL);
    llvm::FunctionType *llvmFunctionType = functionType->LLVMFunctionType(g->ctx, disableMask);
    if (llvmFunctionType == nullptr)
        return;

    // According to the LLVM philosophy, function declaration itself can't have internal linkage.
    // Function may have internal linkage only if it is defined in the module.
    // So here we set external linkage for all function declarations. It will be changed to internal
    // if the function is defined in the module in Function::GenerateIR().
    llvm::GlobalValue::LinkageTypes linkage = llvm::GlobalValue::ExternalLinkage;

    auto [name_pref, name_suf] = functionType->GetFunctionMangledName(false);
    std::string functionName = name_pref + name + name_suf;

    llvm::Function *function = llvm::Function::Create(llvmFunctionType, linkage, functionName.c_str(), module);

    if (g->target_os == TargetOS::windows) {
        // Make export functions callable from DLLs.
        if ((g->dllExport) && (storageClass != SC_STATIC)) {
            function->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
        }
    }

    if (isNoInline && isInline) {
        Error(pos, "Illegal to use \"noinline\" and \"inline\" qualifiers together on function \"%s\".", name.c_str());
        return;
    }
    // Set function attributes: we never throw exceptions
    function->setDoesNotThrow();
    if ((storageClass != SC_EXTERN_C) && (storageClass != SC_EXTERN_SYCL) && isInline) {
        function->addFnAttr(llvm::Attribute::AlwaysInline);
    }

    if (isVectorCall) {
        if ((storageClass != SC_EXTERN_C)) {
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
        if ((storageClass != SC_EXTERN_C) && (storageClass != SC_EXTERN_SYCL)) {
            Error(pos, "Illegal to use \"__regcall\" qualifier on non-extern function \"%s\".", name.c_str());
            return;
        }
    }

    if (isNoInline) {
        function->addFnAttr(llvm::Attribute::NoInline);
    }

    AddUWTableFuncAttr(function);

    if (functionType->isTask) {
        if (!g->target->isXeTarget()) {
            // This also applies transitively to members I think?
            function->addParamAttr(0, llvm::Attribute::NoAlias);
        }
    }
    function->setCallingConv(functionType->GetCallingConv());
    g->target->markFuncWithTargetAttr(function);

    // Make sure that the return type isn't 'varying' or vector typed if
    // the function is 'export'ed.
    if (functionType->isExported &&
        lRecursiveCheckValidParamType(functionType->GetReturnType(), false, false, name, pos) == false)
        Error(pos,
              "Illegal to return a \"varying\" or vector type from "
              "exported function \"%s\"",
              name.c_str());

    if (functionType->isTask && functionType->GetReturnType()->IsVoidType() == false)
        Error(pos, "Task-qualified functions must have void return type.");

    if (functionType->isExported || functionType->isExternC || functionType->isExternSYCL ||
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
        if (functionType->isExported)
            lCheckExportedParameterTypes(argType, argName, argPos);
#ifdef ISPC_XE_ENABLED
        if (functionType->IsISPCKernel())
            lCheckTaskParameterTypes(argType, argName, argPos);
#endif

        // ISPC assumes that no pointers alias.  (It should be possible to
        // specify when this is not the case, but this should be the
        // default.)  Set parameter attributes accordingly.  (Only for
        // uniform pointers, since varying pointers are int vectors...)
        if (!functionType->isTask && !functionType->isExternSYCL &&
            ((CastType<PointerType>(argType) != nullptr && argType->IsUniformType() &&
              // Exclude SOA argument because it is a pair {struct *, int}
              // instead of pointer
              !CastType<PointerType>(argType)->IsSlice()) ||

             CastType<ReferenceType>(argType) != nullptr)) {

            function->addParamAttr(i, llvm::Attribute::NoAlias);
        }

        if (symbolTable->LookupFunction(argName.c_str())) {
            Warning(argPos, "Function parameter \"%s\" shadows a function declared in global scope.", argName.c_str());
        }

        if (symbolTable->LookupFunctionTemplate(argName.c_str())) {
            Warning(argPos, "Function parameter \"%s\" shadows a function template declared in global scope.",
                    argName.c_str());
        }

        if (defaultValue != nullptr)
            seenDefaultArg = true;
        else if (seenDefaultArg) {
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
    Symbol *funSym = new Symbol(name, pos, functionType, storageClass);
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
        TemplateInstantiation inst(*(templateSymbol->templateParms), normTypes, TemplateInstantiationKind::Implicit,
                                   templateSymbol->isInline, templateSymbol->isNoInline);
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
        if (templ->GetStorageClass() == SC_NONE && sc != SC_NONE) {
            Error(pos, "Template instantiation has inconsistent storage class. Consider assigning it to the primary "
                       "template to inherit it's signature.");
            return;
        }
        // If primary template has non-default storage class, but explicit instantiation has different non-default
        // storage class, report an error
        if (templ->GetStorageClass() != SC_NONE && sc != SC_NONE && sc != templ->GetStorageClass()) {
            Error(pos, "Template instantiation has inconsistent storage class.");
            return;
        }
        // If primary template doesn't have unmasked specifier, but explicit instantiation has it,
        // report an error
        if (!templ->GetFunctionType()->isUnmasked && ftype->isUnmasked) {
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
    bool instUnmasked = instType ? instType->isUnmasked : false;
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
    if (templ->GetStorageClass() == SC_NONE && sc != SC_NONE) {
        Error(pos, "Template specialization has inconsistent storage class. Consider assigning it to the primary "
                   "template to inherit it's signature.");
        return;
    }
    // If primary template has non-default storage class, but specialization has different non-default storage class,
    // report an error
    if (templ->GetStorageClass() != SC_NONE && sc != SC_NONE && sc != templ->GetStorageClass()) {
        Error(pos, "Template specialization has inconsistent storage class.");
        return;
    }
    // If primary template doesn't have unmasked specifier, but specialization has it,
    // report an error
    if (!templ->GetFunctionType()->isUnmasked && ftype->isUnmasked) {
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
            CastType<EnumType>(types[i].first) == nullptr)
            Error(types[i].second, "Only struct, vector, and enum types, not \"%s\", are allowed in type export lists.",
                  types[i].first->GetString().c_str());
        else
            exportedTypes.push_back(types[i]);
    }
}

bool Module::writeOutput(OutputType outputType, OutputFlags flags, const char *outFileName,
                         const char *depTargetFileName, const char *sourceFileName, DispatchHeaderInfo *DHI) {
    if (diBuilder && (outputType != Header) && (outputType != Deps))
        lStripUnusedDebugInfo(module);

    Assert(module);

    // In LLVM_3_4 after r195494 and r195504 revisions we should pass
    // "Debug Info Version" constant to the module. LLVM will ignore
    // our Debug Info metadata without it.
    if (g->generateDebuggingSymbols == true) {
        module->addModuleFlag(llvm::Module::Warning, "Debug Info Version", llvm::DEBUG_METADATA_VERSION);
    }

    // SIC! (verifyModule() == TRUE) means "failed", see llvm-link code.
    if ((outputType != Header) && (outputType != Deps) && (outputType != HostStub) && (outputType != DevStub) &&
        (outputType != CPPStub) && llvm::verifyModule(*module)) {
        FATAL("Resulting module verification failed!");
    }

    if (outFileName) {
        // First, issue a warning if the output file suffix and the type of
        // file being created seem to mismatch.  This can help catch missing
        // command-line arguments specifying the output file type.
        const char *suffix = strrchr(outFileName, '.');
        if (suffix != nullptr) {
            ++suffix;
            const char *fileType = nullptr;
            switch (outputType) {
            case Asm:
                if (strcasecmp(suffix, "s"))
                    fileType = "assembly";
                break;
            case Bitcode:
                if (strcasecmp(suffix, "bc"))
                    fileType = "LLVM bitcode";
                break;
            case BitcodeText:
                if (strcasecmp(suffix, "ll"))
                    fileType = "LLVM assembly";
                break;
            case Object:
                if (strcasecmp(suffix, "o") && strcasecmp(suffix, "obj"))
                    fileType = "object";
                break;
#ifdef ISPC_XE_ENABLED
            case ZEBIN:
                if (strcasecmp(suffix, "bin"))
                    fileType = "L0 binary";
                break;
            case SPIRV:
                if (strcasecmp(suffix, "spv"))
                    fileType = "spir-v";
                break;
#endif
            case Header:
                if (strcasecmp(suffix, "h") && strcasecmp(suffix, "hh") && strcasecmp(suffix, "hpp"))
                    fileType = "header";
                break;
            case Deps:
                break;
            case DevStub:
                if (strcasecmp(suffix, "c") && strcasecmp(suffix, "cc") && strcasecmp(suffix, "c++") &&
                    strcasecmp(suffix, "cxx") && strcasecmp(suffix, "cpp"))
                    fileType = "dev-side offload stub";
                break;
            case HostStub:
                if (strcasecmp(suffix, "c") && strcasecmp(suffix, "cc") && strcasecmp(suffix, "c++") &&
                    strcasecmp(suffix, "cxx") && strcasecmp(suffix, "cpp"))
                    fileType = "host-side offload stub";
                break;
            case CPPStub:
                if (strcasecmp(suffix, "ispi") && strcasecmp(suffix, "i"))
                    fileType = "preprocessed stub";
                break;
            default:
                Assert(0 /* swtich case not handled */);
                return 1;
            }
            if (fileType != nullptr)
                Warning(SourcePos(), "Emitting %s file, but filename \"%s\" has suffix \"%s\"?", fileType, outFileName,
                        suffix);
        }
    }
    if (outputType == Header) {
        if (DHI)
            return writeDispatchHeader(DHI);
        else
            return writeHeader(outFileName);
    } else if (outputType == Deps)
        return writeDeps(outFileName, flags.isMakeRuleDeps(), depTargetFileName, sourceFileName);
    else if (outputType == CPPStub)
        return writeCPPStub(outFileName);
    else if (outputType == HostStub)
        return writeHostStub(outFileName);
    else if (outputType == DevStub)
        return writeDevStub(outFileName);
    else if ((outputType == Bitcode) || (outputType == BitcodeText))
        return writeBitcode(module, outFileName, outputType);
#ifdef ISPC_XE_ENABLED
    else if (outputType == SPIRV)
        return writeSPIRV(module, outFileName);
    else if (outputType == ZEBIN)
        return writeZEBin(module, outFileName);
#endif
    else
        return writeObjectFileOrAssembly(outputType, outFileName);
}

bool Module::writeBitcode(llvm::Module *module, const char *outFileName, OutputType outputType) {
    // Get a file descriptor corresponding to where we want the output to
    // go.  If we open it, it'll be closed by the llvm::raw_fd_ostream
    // destructor.
    int fd;
    Assert(outFileName);
    if (!strcmp(outFileName, "-"))
        fd = 1; // stdout
    else {
        int flags = O_CREAT | O_WRONLY | O_TRUNC;
#ifdef ISPC_HOST_IS_WINDOWS
        flags |= O_BINARY;
        fd = _open(outFileName, flags, 0644);
#else
        fd = open(outFileName, flags, 0644);
#endif // ISPC_HOST_IS_WINDOWS
        if (fd == -1) {
            perror(outFileName);
            return false;
        }
    }

    llvm::raw_fd_ostream fos(fd, (fd != 1), false);
    if (outputType == Bitcode)
        llvm::WriteBitcodeToFile(*module, fos);
    else if (outputType == BitcodeText)
        module->print(fos, nullptr);
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

bool Module::writeSPIRV(llvm::Module *module, const char *outFileName) {
    std::stringstream translatedStream;
    bool success = translateToSPIRV(module, translatedStream);
    if (!success) {
        return false;
    }
    if (!strcmp(outFileName, "-")) {
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
        return name.endswith(requiredExtension);
    });
    Assert(binIt != zip.end() && "Output binary is missing");

    llvm::ArrayRef<uint8_t> binRef{std::get<0>(*binIt), static_cast<std::size_t>(std::get<1>(*binIt))};
    oclocRes.assign(binRef.begin(), binRef.end());
}

bool Module::writeZEBin(llvm::Module *module, const char *outFileName) {
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

    if (!strcmp(outFileName, "-")) {
        std::cout.write(oclocRes.data(), oclocRes.size());
    } else {
        std::ofstream fos(outFileName, std::ios::binary);
        fos.write(oclocRes.data(), oclocRes.size());
    }
    return true;
}
#endif // ISPC_XE_ENABLED

bool Module::writeObjectFileOrAssembly(OutputType outputType, const char *outFileName) {
    llvm::TargetMachine *targetMachine = g->target->GetTargetMachine();
    return writeObjectFileOrAssembly(targetMachine, module, outputType, outFileName);
}

bool Module::writeObjectFileOrAssembly(llvm::TargetMachine *targetMachine, llvm::Module *module, OutputType outputType,
                                       const char *outFileName) {
    // Figure out if we're generating object file or assembly output, and
    // set binary output for object files
#if ISPC_LLVM_VERSION > ISPC_LLVM_17_0
    llvm::CodeGenFileType fileType =
        (outputType == Object) ? llvm::CodeGenFileType::ObjectFile : llvm::CodeGenFileType::AssemblyFile;
    bool binary = (fileType == llvm::CodeGenFileType::ObjectFile);
#else
    llvm::CodeGenFileType fileType = (outputType == Object) ? llvm::CGFT_ObjectFile : llvm::CGFT_AssemblyFile;
    bool binary = (fileType == llvm::CGFT_ObjectFile);

#endif
    llvm::sys::fs::OpenFlags flags = binary ? llvm::sys::fs::OF_None : llvm::sys::fs::OF_Text;

    std::error_code error;

    std::unique_ptr<llvm::ToolOutputFile> of(new llvm::ToolOutputFile(outFileName, error, flags));

    if (error) {
        Error(SourcePos(), "Cannot open output file \"%s\".\n", outFileName);
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
        pm.run(*module);

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
    if (st != nullptr)
        return st;

    const ArrayType *at = CastType<ArrayType>(t);
    if (at != nullptr)
        return lGetElementStructType(at->GetElementType());

    return nullptr;
}

static bool lContainsPtrToVarying(const StructType *st) {
    int numElts = st->GetElementCount();

    for (int j = 0; j < numElts; ++j) {
        const Type *t = st->GetElementType(j);

        if (t->IsVaryingType())
            return true;
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
    for (int i = 0; i < (int)emittedStructs->size(); ++i)
        if (Type::EqualIgnoringConst(st, (*emittedStructs)[i]))
            return;

    // Otherwise first make sure any contained structs have been declared.
    for (int i = 0; i < st->GetElementCount(); ++i) {
        const StructType *elementStructType = lGetElementStructType(st->GetElementType(i));
        if (elementStructType != nullptr)
            lEmitStructDecl(elementStructType, emittedStructs, file, emitUnifs);
    }

    // And now it's safe to declare this one
    emittedStructs->push_back(st);

    fprintf(file, "#ifndef __ISPC_STRUCT_%s__\n", st->GetCStructName().c_str());
    fprintf(file, "#define __ISPC_STRUCT_%s__\n", st->GetCStructName().c_str());

    char sSOA[48];
    bool pack, needsAlign = false;
    llvm::Type *stype = st->LLVMType(g->ctx);
    const llvm::DataLayout *DL = g->target->getDataLayout();

    llvm::StructType *stypeStructType = llvm::dyn_cast<llvm::StructType>(stype);
    Assert(stypeStructType);
    if (!(pack = stypeStructType->isPacked())) {
        for (int i = 0; !needsAlign && (i < st->GetElementCount()); ++i) {
            const Type *ftype = st->GetElementType(i)->GetAsNonConstType();
            needsAlign |= ftype->IsVaryingType() && (CastType<StructType>(ftype) == nullptr);
        }
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
        unsigned uABI = DL->getABITypeAlign(stype).value();
        fprintf(file, "__ISPC_ALIGNED_STRUCT__(%u) %s%s {\n", uABI, st->GetCStructName().c_str(), sSOA);
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

    fprintf(file, "\n#ifndef __ISPC_ALIGN__\n"
                  "#if defined(__clang__) || !defined(_MSC_VER)\n"
                  "// Clang, GCC, ICC\n"
                  "#define __ISPC_ALIGN__(s) __attribute__((aligned(s)))\n"
                  "#define __ISPC_ALIGNED_STRUCT__(s) struct __ISPC_ALIGN__(s)\n"
                  "#else\n"
                  "// Visual Studio\n"
                  "#define __ISPC_ALIGN__(s) __declspec(align(s))\n"
                  "#define __ISPC_ALIGNED_STRUCT__(s) __ISPC_ALIGN__(s) struct\n"
                  "#endif\n"
                  "#endif\n\n");

    for (unsigned int i = 0; i < structTypes.size(); ++i)
        lEmitStructDecl(structTypes[i], &emittedStructs, file, emitUnifs);
}

/** Emit C declarations of enumerator types to the generated header file.
 */
static void lEmitEnumDecls(const std::vector<const EnumType *> &enumTypes, FILE *file) {
    if (enumTypes.size() == 0)
        return;

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
    if (types.size() == 0)
        return;

    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n");
    fprintf(file, "// Vector types with external visibility from ispc code\n");
    fprintf(file, "///////////////////////////////////////////////////////////////////////////\n\n");

    for (unsigned int i = 0; i < types.size(); ++i) {
        std::string baseDecl;
        const VectorType *vt = types[i]->GetAsNonConstType();
        if (!vt->IsUniformType())
            // Varying stuff shouldn't be visibile to / used by the
            // application, so at least make it not simple to access it by
            // not declaring the type here...
            continue;

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
    for (unsigned int i = 0; i < exportedTypes->size(); ++i)
        if (Type::Equal((*exportedTypes)[i], type))
            return;

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

    if (CastType<ReferenceType>(type) != nullptr)
        lGetExportedTypes(type->GetReferenceTarget(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
    else if (CastType<PointerType>(type) != nullptr)
        lGetExportedTypes(type->GetBaseType(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
    else if (arrayType != nullptr)
        lGetExportedTypes(arrayType->GetElementType(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
    else if (structType != nullptr) {
        lAddTypeIfNew(type, exportedStructTypes);
        for (int i = 0; i < structType->GetElementCount(); ++i)
            lGetExportedTypes(structType->GetElementType(i), exportedStructTypes, exportedEnumTypes,
                              exportedVectorTypes);
    } else if (CastType<UndefinedStructType>(type) != nullptr)
        // do nothing
        ;
    else if (CastType<EnumType>(type) != nullptr)
        lAddTypeIfNew(type, exportedEnumTypes);
    else if (CastType<VectorType>(type) != nullptr)
        lAddTypeIfNew(type, exportedVectorTypes);
    else if (ftype != nullptr) {
        // Handle Return Types
        lGetExportedTypes(ftype->GetReturnType(), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);

        // And now the parameter types...
        for (int j = 0; j < ftype->GetNumParameters(); ++j)
            lGetExportedTypes(ftype->GetParameterType(j), exportedStructTypes, exportedEnumTypes, exportedVectorTypes);
    } else
        Assert(CastType<AtomicType>(type) != nullptr);
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
    if (useExternC)
        fprintf(file, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\nextern "
                      "\"C\" {\n#endif // __cplusplus\n");
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
    if (useExternC)

        fprintf(file, "#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )\n} /* end "
                      "extern C */\n#endif // __cplusplus\n");
    // fprintf(file, "#ifdef __cplusplus\n} /* end extern C */\n#endif // __cplusplus\n");
}

static bool lIsExported(const Symbol *sym) {
    const FunctionType *ft = CastType<FunctionType>(sym->type);
    Assert(ft);
    return ft->isExported;
}

static bool lIsExternC(const Symbol *sym) {
    const FunctionType *ft = CastType<FunctionType>(sym->type);
    Assert(ft);
    return ft->isExternC;
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

bool Module::writeDeps(const char *fn, bool generateMakeRule, const char *tn, const char *sn) {
    if (fn && g->debugPrint) { // We may be passed nullptr for stdout output.
        printf("\nWriting dependencies to file %s\n", fn);
    }
    FILE *file = fn ? fopen(fn, "w") : stdout;
    if (!file) {
        perror("fopen");
        return false;
    }

    if (generateMakeRule) {
        Assert(tn);
        fprintf(file, "%s:", tn);
        // Rules always emit source first.
        if (sn && !IsStdin(sn)) {
            fprintf(file, " %s", sn);
        }
        std::string unescaped;

        for (std::set<std::string>::const_iterator it = registeredDependencies.begin();
             it != registeredDependencies.end(); ++it) {
            unescaped = *it; // As this is preprocessor output, paths come escaped.
            lUnescapeStringInPlace(unescaped);
            if (sn && !IsStdin(sn) &&
                0 == strcmp(sn, unescaped.c_str())) // If source has been passed, it's already emitted.
                continue;
            fprintf(file, " \\\n");
            fprintf(file, " %s", unescaped.c_str());
        }
        fprintf(file, "\n");
    } else {
        for (std::set<std::string>::const_iterator it = registeredDependencies.begin();
             it != registeredDependencies.end(); ++it)
            fprintf(file, "%s\n", it->c_str());
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
        const Type *paramType;
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
        std::string paramTypeName = paramType->GetString();

        std::string tmpArgDecl = paramType->GetDeclaration(paramName, DeclarationSyntax::CPP);
        out << "   " << tmpArgDecl << ";" << std::endl;
    }

    out << "};" << std::endl;
    return out.str();
}

bool Module::writeDevStub(const char *fn) {
    FILE *file = fopen(fn, "w");
    if (!file) {
        perror("fopen");
        return false;
    }
    fprintf(file, "//\n// %s\n// (device stubs automatically generated by the ispc compiler.)\n", fn);
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
            // const Type *paramType = fct->GetParameterType(i)->GetAsNonConstType();
            const Type *paramType; // = fct->GetParameterType(i)->GetAsNonConstType();
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
            std::string paramTypeName = paramType->GetString();

            if (i)
                funcall << ", ";
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

bool Module::writeCPPStub(const char *outFileName) { return writeCPPStub(this, outFileName); }

bool Module::writeCPPStub(Module *module, const char *outFileName) {
    // Get a file descriptor corresponding to where we want the output to
    // go.  If we open it, it'll be closed by the llvm::raw_fd_ostream
    // destructor.
    int fd;
    int flags = O_CREAT | O_WRONLY | O_TRUNC;

    if (!outFileName) {
        return false;
    } else if (!strcmp("-", outFileName)) {
        fd = 1;
    } else {
#ifdef ISPC_HOST_IS_WINDOWS
        fd = _open(outFileName, flags, 0644);
#else
        fd = open(outFileName, flags, 0644);
#endif // ISPC_HOST_IS_WINDOWS
        if (fd == -1) {
            perror(outFileName);
            return false;
        }
    }

    // The CPP stream should have been initialized: print and clean it up.
    Assert(module->bufferCPP && "`bufferCPP` should not be null");
    llvm::raw_fd_ostream fos(fd, (fd != 1), false);
    fos << module->bufferCPP->str;
    module->bufferCPP.reset();

    return true;
}

bool Module::writeHostStub(const char *fn) {
    FILE *file = fopen(fn, "w");
    if (!file) {
        perror("fopen");
        return false;
    }
    fprintf(file, "//\n// %s\n// (device stubs automatically generated by the ispc compiler.)\n", fn);
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
                if (numPointers)
                    pointerArgs << ",";
                pointerArgs << "(void*)" << paramName;
                numPointers++;
                continue;
            }

            fprintf(file, "  __args.%s = %s;\n", paramName.c_str(), paramName.c_str());
        }
        // ------------------------------------------------------------------
        // writer pointer list
        // ------------------------------------------------------------------
        if (numPointers == 0)
            pointerArgs << "NULL";
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

bool Module::writeHeader(const char *fn) {
    FILE *f = fopen(fn, "w");
    if (!f) {
        perror("fopen");
        return false;
    }
    fprintf(f, "//\n// %s\n// (Header automatically generated by the ispc compiler.)\n", fn);
    fprintf(f, "// DO NOT EDIT THIS FILE.\n//\n\n");

    // Create a nice guard string from the filename, turning any
    // non-number/letter characters into underbars
    std::string guard = "ISPC_";
    const char *p = fn;
    while (*p) {
        if (isdigit(*p))
            guard += *p;
        else if (isalpha(*p))
            guard += toupper(*p);
        else
            guard += "_";
        ++p;
    }

    if (g->noPragmaOnce)
        fprintf(f, "#ifndef %s\n#define %s\n\n", guard.c_str(), guard.c_str());
    else
        fprintf(f, "#pragma once\n");

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
        if (const StructType *st = CastType<StructType>(exportedTypes[i].first))
            exportedStructTypes.push_back(st->GetAsUniformType());
        else if (const EnumType *et = CastType<EnumType>(exportedTypes[i].first))
            exportedEnumTypes.push_back(et->GetAsUniformType());
        else if (const VectorType *vt = CastType<VectorType>(exportedTypes[i].first))
            exportedVectorTypes.push_back(vt->GetAsUniformType());
        else
            FATAL("Unexpected type in export list");
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
    if (g->noPragmaOnce)
        fprintf(f, "\n#endif // %s\n", guard.c_str());

    fclose(f);
    return true;
}

struct ispc::DispatchHeaderInfo {
    bool EmitUnifs;
    bool EmitFuncs;
    bool EmitFrontMatter;
    bool EmitBackMatter;
    bool Emit4;
    bool Emit8;
    bool Emit16;
    FILE *file;
    const char *fn;
};

bool Module::writeDispatchHeader(DispatchHeaderInfo *DHI) {
    FILE *f = DHI->file;

    if (DHI->EmitFrontMatter) {
        fprintf(f, "//\n// %s\n// (Header automatically generated by the ispc compiler.)\n", DHI->fn);
        fprintf(f, "// DO NOT EDIT THIS FILE.\n//\n\n");
    }
    // Create a nice guard string from the filename, turning any
    // non-number/letter characters into underbars
    std::string guard = "ISPC_";
    const char *p = DHI->fn;
    while (*p) {
        if (isdigit(*p))
            guard += *p;
        else if (isalpha(*p))
            guard += toupper(*p);
        else
            guard += "_";
        ++p;
    }
    if (DHI->EmitFrontMatter) {
        if (g->noPragmaOnce)
            fprintf(f, "#ifndef %s\n#define %s\n\n", guard.c_str(), guard.c_str());
        else
            fprintf(f, "#pragma once\n");

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

        // Go through the explicitly exported types
        for (int i = 0; i < (int)exportedTypes.size(); ++i) {
            if (const StructType *st = CastType<StructType>(exportedTypes[i].first))
                exportedStructTypes.push_back(st->GetAsUniformType());
            else if (const EnumType *et = CastType<EnumType>(exportedTypes[i].first))
                exportedEnumTypes.push_back(et->GetAsUniformType());
            else if (const VectorType *vt = CastType<VectorType>(exportedTypes[i].first))
                exportedVectorTypes.push_back(vt->GetAsUniformType());
            else
                FATAL("Unexpected type in export list");
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
        if (g->noPragmaOnce)
            fprintf(f, "\n#endif // %s\n", guard.c_str());
        DHI->EmitBackMatter = false;
    }
    return true;
}

// Copied and reduced from CompilerInstance::InitializeSourceManager to avoid dependencies from CompilerInstance
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
        if (inputFile != "-")
            diag.Report(clang::diag::err_fe_error_reading) << inputFile;
        else
            diag.Report(clang::diag::err_fe_error_reading_stdin) << errCode.message();
        return;
    }

    srcMgr.setMainFileID(srcMgr.createFileID(*fileOrError, clang::SourceLocation(), kind));
    Assert(srcMgr.getMainFileID().isValid() && "Couldn't establish MainFileID");
    return;
}

// Copied from InitPreprocessor.cpp
static bool lMacroBodyEndsInBackslash(llvm::StringRef MacroBody) {
    while (!MacroBody.empty() && clang::isWhitespace(MacroBody.back()))
        MacroBody = MacroBody.drop_back();
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
        if (End != llvm::StringRef::npos)
            Diags.Report(clang::diag::warn_fe_macro_contains_embedded_newline) << MacroName;
        MacroBody = MacroBody.substr(0, End);
        // We handle macro bodies which end in a backslash by appending an extra
        // backslash+newline.  This makes sure we don't accidentally treat the
        // backslash as a line continuation marker.
        if (lMacroBodyEndsInBackslash(MacroBody))
            Builder.defineMacro(MacroName, llvm::Twine(MacroBody) + "\\\n");
        else
            Builder.defineMacro(MacroName, MacroBody);
    } else {
        // Push "macroname 1".
        Builder.defineMacro(Macro);
    }
}

// Core logic of buiding macros copied from clang::InitializePreprocessor from InitPreprocessor.cpp
static void lInitializePreprocessor(clang::Preprocessor &PP, const clang::PreprocessorOptions &InitOpts) {
    std::string PredefineBuffer;
    PredefineBuffer.reserve(4080);
    llvm::raw_string_ostream Predefines(PredefineBuffer);
    clang::MacroBuilder Builder(Predefines);

    // Process #define's and #undef's in the order they are given.
    for (unsigned i = 0, e = InitOpts.Macros.size(); i != e; ++i) {
        if (InitOpts.Macros[i].second) // isUndef
            Builder.undefineMacro(InitOpts.Macros[i].first);
        else
            lDefineBuiltinMacro(Builder, InitOpts.Macros[i].first, PP.getDiagnostics());
    }

    // Copy PredefinedBuffer into the Preprocessor.
    PP.setPredefines(std::move(PredefineBuffer));
}

static void lSetPreprocessorOutputOptions(clang::PreprocessorOutputOptions *opts) {
    // Don't remove comments in the preprocessor, so that we can accurately
    // track the source file position by handling them ourselves.
    opts->ShowComments = 1;
    opts->ShowCPP = 1;
}

static void lSetHeaderSeachOptions(const std::shared_ptr<clang::HeaderSearchOptions> opts) {
    opts->UseBuiltinIncludes = 0;
    opts->UseStandardSystemIncludes = 0;
    opts->UseStandardCXXIncludes = 0;
    if (g->debugPrint)
        opts->Verbose = 1;
    for (int i = 0; i < (int)g->includePath.size(); ++i) {
        opts->AddPath(g->includePath[i], clang::frontend::Angled, false /* not a framework */,
                      true /* ignore sys root */);
    }
}

static void lSetPreprocessorOptions(const std::shared_ptr<clang::PreprocessorOptions> opts) {
    // Add defs for ISPC and PI
    opts->addMacroDef("ISPC");
    opts->addMacroDef("PI=3.1415926535");

    // Add definitions of limits for integers and float types.
    opts->addMacroDef("INT8_MIN=-128");
    opts->addMacroDef("INT8_MAX=127");
    opts->addMacroDef("UINT8_MAX=255U");
    opts->addMacroDef("INT16_MIN=-32768");
    opts->addMacroDef("INT16_MAX=32767");
    opts->addMacroDef("UINT16_MAX=65535U");
    opts->addMacroDef("INT32_MIN=-2147483648L");
    opts->addMacroDef("INT32_MAX=2147483647L");
    opts->addMacroDef("UINT32_MAX=4294967295UL");
    opts->addMacroDef("INT64_MIN=-9223372036854775808LL");
    opts->addMacroDef("INT64_MAX=9223372036854775807LL");
    opts->addMacroDef("UINT64_MAX=18446744073709551615ULL");
    opts->addMacroDef("F16_MIN=6.103515625e-05F16");
    opts->addMacroDef("F16_MAX=65504.0F16");
    opts->addMacroDef("FLT_MIN=1.17549435082228750796873653722224568e-38F");
    opts->addMacroDef("FLT_MAX=3.40282346638528859811704183484516925e+38F");
    opts->addMacroDef("DBL_MIN=2.22507385850720138309023271733240406e-308D");
    opts->addMacroDef("DBL_MAX=1.79769313486231570814527423731704357e+308D");

    if (g->enableLLVMIntrinsics) {
        opts->addMacroDef("ISPC_LLVM_INTRINSICS_ENABLED");
    }
    // Add defs for ISPC_UINT_IS_DEFINED.
    // This lets the user know uint* is part of language.
    opts->addMacroDef("ISPC_UINT_IS_DEFINED");

    // Add #define for current compilation target
    char targetMacro[128];
    snprintf(targetMacro, sizeof(targetMacro), "ISPC_TARGET_%s", g->target->GetISAString());
    char *p = targetMacro;
    while (*p) {
        *p = toupper(*p);
        if (*p == '-')
            *p = '_';
        ++p;
    }

    // Add 'TARGET_WIDTH' macro to expose vector width to user.
    std::string TARGET_WIDTH = "TARGET_WIDTH=" + std::to_string(g->target->getVectorWidth());
    opts->addMacroDef(TARGET_WIDTH);

    // Add 'TARGET_ELEMENT_WIDTH' macro to expose element width to user.
    std::string TARGET_ELEMENT_WIDTH = "TARGET_ELEMENT_WIDTH=" + std::to_string(g->target->getDataTypeWidth() / 8);
    opts->addMacroDef(TARGET_ELEMENT_WIDTH);

    opts->addMacroDef(targetMacro);

    if (g->target->is32Bit())
        opts->addMacroDef("ISPC_POINTER_SIZE=32");
    else
        opts->addMacroDef("ISPC_POINTER_SIZE=64");

    if (g->target->hasHalfConverts())
        opts->addMacroDef("ISPC_TARGET_HAS_HALF");
    if (g->target->hasRand())
        opts->addMacroDef("ISPC_TARGET_HAS_RAND");
    if (g->target->hasTranscendentals())
        opts->addMacroDef("ISPC_TARGET_HAS_TRANSCENDENTALS");
    if (g->opt.forceAlignedMemory)
        opts->addMacroDef("ISPC_FORCE_ALIGNED_MEMORY");

    constexpr int buf_size = 25;
    char ispc_major[buf_size], ispc_minor[buf_size];
    snprintf(ispc_major, buf_size, "ISPC_MAJOR_VERSION=%d", ISPC_VERSION_MAJOR);
    snprintf(ispc_minor, buf_size, "ISPC_MINOR_VERSION=%d", ISPC_VERSION_MINOR);
    opts->addMacroDef(ispc_major);
    opts->addMacroDef(ispc_minor);

    if (g->target->hasFp16Support()) {
        opts->addMacroDef("ISPC_FP16_SUPPORTED ");
    }

    if (g->target->hasFp64Support()) {
        opts->addMacroDef("ISPC_FP64_SUPPORTED ");
    }

    if (g->includeStdlib) {
        if (g->opt.disableAsserts)
            opts->addMacroDef("assert(x)=");
        else
            opts->addMacroDef("assert(x)=__assert(#x, x)");
    }

    for (unsigned int i = 0; i < g->cppArgs.size(); ++i) {
        // Sanity check--should really begin with -D
        if (g->cppArgs[i].substr(0, 2) == "-D") {
            opts->addMacroDef(g->cppArgs[i].substr(2));
        }
    }
}

static void lSetLangOptions(clang::LangOptions *opts) { opts->LineComment = 1; }

int Module::execPreprocessor(const char *infilename, llvm::raw_string_ostream *ostream) const {
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
    lSetPreprocessorOutputOptions(&preProcOutOpts);

    // Create and initialize HeaderSearchOptions
    const std::shared_ptr<clang::HeaderSearchOptions> hdrSearchOpts = std::make_shared<clang::HeaderSearchOptions>();
    lSetHeaderSeachOptions(hdrSearchOpts);

    // Create and initializer PreprocessorOptions
    const std::shared_ptr<clang::PreprocessorOptions> preProcOpts = std::make_shared<clang::PreprocessorOptions>();
    lSetPreprocessorOptions(preProcOpts);

    // Create and initializer LangOptions
    clang::LangOptions langOpts;
    lSetLangOptions(&langOpts);

    // Create and initialize SourceManager
    clang::FileSystemOptions fsOpts;
    clang::FileManager fileMgr(fsOpts);
    clang::SourceManager srcMgr(diagEng, fileMgr);
    lInitializeSourceManager(inputFile, diagEng, fileMgr, srcMgr);

    // Create HeaderSearch and apply HeaderSearchOptions
    clang::HeaderSearch hdrSearch(hdrSearchOpts, srcMgr, diagEng, langOpts, tgtInfo);
    clang::ApplyHeaderSearchOptions(hdrSearch, *hdrSearchOpts, langOpts, triple);

    // Finally, create an preprocessor object
    clang::TrivialModuleLoader modLoader;
    clang::Preprocessor prep(preProcOpts, diagEng, langOpts, srcMgr, hdrSearch, modLoader);

    // intialize preprocessor
    prep.Initialize(*tgtInfo);
    prep.setPreprocessedOutput(preProcOutOpts.ShowCPP);
    lInitializePreprocessor(prep, *preProcOpts);

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

// Given an output filename of the form "foo.obj", and an ISA name like
// "avx", return a string with the ISA name inserted before the original
// filename's suffix, like "foo_avx.obj".
static std::string lGetTargetFileName(const char *outFileName, const std::string &isaString) {
    assert(outFileName != nullptr && "`outFileName` should not be null");

    std::string targetOutFileName{outFileName};
    const auto pos_dot = targetOutFileName.find_last_of('.');

    if (pos_dot != std::string::npos) {
        targetOutFileName = targetOutFileName.substr(0, pos_dot) + "_" + isaString + targetOutFileName.substr(pos_dot);
    } else {
        // Can't find a '.' in the filename, so just append the ISA suffix
        // to what we weregiven
        targetOutFileName.append("_" + isaString);
    }
    return targetOutFileName;
}

static bool lSymbolIsExported(const Symbol *s) { return s->exportedFunction != nullptr; }

// Small structure to hold pointers to the various different versions of a
// llvm::Function that were compiled for different compilation target ISAs.
struct FunctionTargetVariants {
    FunctionTargetVariants() {
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
                        ftype[j] = LLVMTypes::Int8PointerType;
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
        } else
            targetFuncs[i] = nullptr;
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
        if (targetFuncs[i] == nullptr)
            continue;

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

    // We couldn't find a match that the current system was capable of
    // running.  We'll call abort(); this is a bit of a blunt hammer--it
    // might be preferable to call a user-supplied callback--ISPCError(...)
    // or some such, but we don't want to start imposing too much of a
    // runtime library requirement either...
    llvm::Function *abortFunc = module->getFunction("abort");
    Assert(abortFunc);
    llvm::CallInst::Create(abortFunc, "", bblock);

    // Return an undef value from the function here; we won't get to this
    // point at runtime, but LLVM needs all of the basic blocks to be
    // terminated...
    if (voidReturn)
        llvm::ReturnInst::Create(*g->ctx, bblock);
    else {
        llvm::Value *undefRet = llvm::UndefValue::get(ftype->getReturnType());
        llvm::ReturnInst::Create(*g->ctx, undefRet, bblock);
    }
}

// Initialize a dispatch module
static llvm::Module *lInitDispatchModule() {
    llvm::Module *module = new llvm::Module("dispatch_module", *g->ctx);

    module->setTargetTriple(g->target->GetTripleString());

    // DataLayout information supposed to be managed in single place in Target class.
    module->setDataLayout(g->target->getDataLayout()->getStringRepresentation());

    // First, link in the definitions from the builtins-dispatch.ll file.
    const BitcodeLib *dispatch = g->target_registry->getDispatchLib(g->target_os);
    Assert(dispatch);
    AddBitcodeToModule(dispatch, module);

    lSetCodeModel(module);
    lSetPICLevel(module);

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
    llvm::Function *setFunc = module->getFunction("__set_system_isa");
    Assert(setFunc != nullptr);
    llvm::Value *systemBestISAPtr = module->getGlobalVariable("__system_best_isa", true);
    Assert(systemBestISAPtr != nullptr);

    // For each exported function, create the dispatch function
    std::map<std::string, FunctionTargetVariants>::iterator iter;
    for (iter = functions.begin(); iter != functions.end(); ++iter)
        lCreateDispatchFunction(module, setFunc, systemBestISAPtr, iter->first, iter->second);

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
    while (Ty1->getTypeID() == Ty2->getTypeID())
        switch (Ty1->getTypeID()) {
        case llvm::Type::ArrayTyID:
            if (Ty1->getArrayNumElements() != Ty2->getArrayNumElements())
                return false;
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
    return false;
}

// Grab all of the global value definitions from the module and change them
// to be declarations; we'll emit a single definition of each global in the
// final module used with the dispatch functions, so that we don't have
// multiple definitions of them, one in each of the target-specific output
// files.
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
            // Turn this into an 'extern' declaration by clearing its
            // initializer.
            gv->setInitializer(nullptr);
        }
    }
}

int Module::CompileAndOutput(const char *srcFile, Arch arch, const char *cpu, std::vector<ISPCTarget> targets,
                             OutputFlags outputFlags, OutputType outputType, const char *outFileName,
                             const char *headerFileName, const char *depsFileName, const char *depsTargetName,
                             const char *hostStubFileName, const char *devStubFileName) {
    if (targets.size() == 0 || targets.size() == 1) {
        // We're only compiling to a single target
        // TODO something wrong here
        ISPCTarget target = ISPCTarget::none;
        if (targets.size() == 1) {
            target = targets[0];
        }
        g->target = new Target(arch, cpu, target, outputFlags.getPICLevel(), outputFlags.getMCModel(), g->printTarget);
        if (!g->target->isValid())
            return 1;

        m = new Module(srcFile);
        const int compileResult = m->CompileFile();

        llvm::TimeTraceScope TimeScope("Backend");

        if (compileResult == 0) {
#ifdef ISPC_XE_ENABLED
            if (outputType == Asm || outputType == Object) {
                if (g->target->isXeTarget()) {
                    Error(SourcePos(), "%s output is not supported yet for Xe targets. ",
                          (outputType == Asm) ? "assembly" : "binary");
                    return 1;
                }
            }
            if (g->target->isXeTarget() && outputType == OutputType::Object) {
                outputType = OutputType::ZEBIN;
            }
            if (!g->target->isXeTarget() && (outputType == OutputType::ZEBIN || outputType == OutputType::SPIRV)) {
                Error(SourcePos(), "SPIR-V and L0 binary formats are supported for Xe target only");
                return 1;
            }
#endif
            if (outFileName != nullptr)
                if (!m->writeOutput(outputType, outputFlags, outFileName))
                    return 1;
            if (headerFileName != nullptr)
                if (!m->writeOutput(Module::Header, outputFlags, headerFileName))
                    return 1;
            if (depsFileName != nullptr || outputFlags.isDepsToStdout()) {
                std::string targetName;
                if (depsTargetName)
                    targetName = depsTargetName;
                else if (outFileName)
                    targetName = outFileName;
                else if (!IsStdin(srcFile)) {
                    targetName = srcFile;
                    size_t dot = targetName.find_last_of('.');
                    if (dot != std::string::npos)
                        targetName.erase(dot, std::string::npos);
                    targetName.append(".o");
                } else
                    targetName = "a.out";
                if (!m->writeOutput(Module::Deps, outputFlags, depsFileName, targetName.c_str(), srcFile))
                    return 1;
            }
            if (hostStubFileName != nullptr)
                if (!m->writeOutput(Module::HostStub, outputFlags, hostStubFileName))
                    return 1;
            if (devStubFileName != nullptr)
                if (!m->writeOutput(Module::DevStub, outputFlags, devStubFileName))
                    return 1;
        } else {
            ++m->errorCount;
        }

        int errorCount = m->errorCount;
        // In case of error, clean up symbolTable
        if (errorCount > 0) {
            m->symbolTable->PopInnerScopes();
        }
        delete m;
        m = nullptr;

        delete g->target;
        g->target = nullptr;

        return errorCount > 0;
    } else {
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

        // The user supplied multiple targets
        Assert(targets.size() > 1);

        if (outFileName != nullptr && strcmp(outFileName, "-") == 0) {
            Error(SourcePos(), "Multi-target compilation can't generate output "
                               "to stdout.  Please provide an output filename.\n");
            return 1;
        }

        // Make sure that the function names for 'export'ed functions have
        // the target ISA appended to them.
        g->mangleFunctionsWithTarget = true;

        // Array initialized with all false
        bool compiledTargets[Target::NUM_ISAS] = {};

        llvm::Module *dispatchModule = nullptr;

        std::map<std::string, FunctionTargetVariants> exportedFunctions;
        int errorCount = 0;

        // Handle creating a "generic" header file for multiple targets
        // that use exported varyings
        DispatchHeaderInfo DHI;
        if (headerFileName != nullptr) {
            DHI.file = fopen(headerFileName, "w");
            if (!DHI.file) {
                perror("fopen");
                return false;
            }
            DHI.fn = headerFileName;
            DHI.EmitUnifs = true;
            DHI.EmitFuncs = true;
            DHI.EmitFrontMatter = true;
            DHI.Emit4 = true;
            DHI.Emit8 = true;
            DHI.Emit16 = true;
            // This is toggled later.
            DHI.EmitBackMatter = false;
        }

        std::vector<Module *> modules(targets.size());
        for (unsigned int i = 0; i < targets.size(); ++i) {
            g->target =
                new Target(arch, cpu, targets[i], outputFlags.getPICLevel(), outputFlags.getMCModel(), g->printTarget);
            if (!g->target->isValid())
                return 1;

            // Issue an error if we've already compiled to a variant of
            // this target ISA.  (It doesn't make sense to compile to both
            // avx and avx-x2, for example.)
            auto targetISA = g->target->getISA();
            if (compiledTargets[targetISA] || (compiledTargets[Target::SSE41] && targetISA == Target::SSE42) ||
                (compiledTargets[Target::SSE42] && targetISA == Target::SSE41)) {
                Error(SourcePos(), "Can't compile to multiple variants of %s target!\n", g->target->GetISAString());
                return 1;
            }
            compiledTargets[targetISA] = true;

            m = new Module(srcFile);
            modules.push_back(m);
            const int compileResult = m->CompileFile();

            llvm::TimeTraceScope TimeScope("Backend");

            if (compileResult == 0) {
                // Create the dispatch module, unless already created;
                // in the latter case, just do the checking
                bool check = (dispatchModule != nullptr);
                if (!check)
                    dispatchModule = lInitDispatchModule();
                lExtractOrCheckGlobals(m->module, dispatchModule, check);

                // Grab pointers to the exported functions from the module we
                // just compiled, for use in generating the dispatch function
                // later.
                lGetExportedFunctions(m->symbolTable, exportedFunctions);

                if (outFileName != nullptr) {
                    std::string targetOutFileName;
                    std::string isaName{g->target->GetISAString()};
                    targetOutFileName = lGetTargetFileName(outFileName, isaName);
                    if (!m->writeOutput(outputType, outputFlags, targetOutFileName.c_str())) {
                        return 1;
                    }
                }
            } else {
                ++m->errorCount;
            }

            errorCount += m->errorCount;
            if (errorCount != 0) {
                return 1;
            }

            // Only write the generate header file, if desired, the first
            // time through the loop here.
            if (headerFileName != nullptr) {
                if (i == targets.size() - 1) {
                    // only print backmatter on the last target.
                    DHI.EmitBackMatter = true;
                }

                const char *isaName;
                isaName = g->target->GetISAString();
                std::string targetHeaderFileName = lGetTargetFileName(headerFileName, isaName);
                // write out a header w/o target name for the first target only
                if (!m->writeOutput(Module::Header, outputFlags, headerFileName, nullptr, nullptr, &DHI)) {
                    return 1;
                }
                if (!m->writeOutput(Module::Header, outputFlags, targetHeaderFileName.c_str())) {
                    return 1;
                }
                if (i == targets.size() - 1) {
                    fclose(DHI.file);
                }
            }

            delete g->target;
            g->target = nullptr;

            // Important: Don't delete the llvm::Module *m here; we need to
            // keep it around so the llvm::Functions *s stay valid for when
            // we generate the dispatch module's functions...
        }

        // Find the first initialized target machine from the targets we
        // compiled to above.  We'll use this as the target machine for
        // compiling the dispatch module--this is safe in that it is the
        // least-common-denominator of all of the targets we compiled to.
        int firstTargetISA = 0;
        while (!compiledTargets[firstTargetISA]) {
            firstTargetISA++;
        }
        const char *firstISA = Target::ISAToTargetString((Target::ISA)firstTargetISA);
        ISPCTarget firstTarget = ParseISPCTarget(firstISA);
        Assert(strcmp(firstISA, "") != 0);
        Assert(firstTarget != ISPCTarget::none);

        g->target = new Target(arch, cpu, firstTarget, outputFlags.getPICLevel(), outputFlags.getMCModel(), false);
        llvm::TargetMachine *firstTargetMachine = g->target->GetTargetMachine();
        Assert(firstTargetMachine);
        if (!g->target->isValid()) {
            return 1;
        }

        if (dispatchModule == nullptr) {
            Error(SourcePos(), "Failed to create dispatch module.\n");
            return 1;
        }

        lEmitDispatchModule(dispatchModule, exportedFunctions);

        if (outFileName != nullptr) {
            switch (outputType) {
            case CPPStub:
                // No preprocessor output for dispatch module.
                break;

            case Bitcode:
            case BitcodeText:
                if (!writeBitcode(dispatchModule, outFileName, outputType))
                    return 1;
                break;

            case Asm:
            case Object:
                if (!writeObjectFileOrAssembly(firstTargetMachine, dispatchModule, outputType, outFileName))
                    return 1;
                break;

            default:
                FATAL("Unexpected `outputType`");
            }
        }

        if (depsFileName != nullptr || outputFlags.isDepsToStdout()) {
            std::string targetName;
            if (depsTargetName)
                targetName = depsTargetName;
            else if (outFileName)
                targetName = outFileName;
            else if (!IsStdin(srcFile)) {
                targetName = srcFile;
                size_t dot = targetName.find_last_of('.');
                if (dot != std::string::npos)
                    targetName.erase(dot, std::string::npos);
                targetName.append(".o");
            } else
                targetName = "a.out";
            if (!m->writeOutput(Module::Deps, outputFlags, depsFileName, targetName.c_str(), srcFile))
                return 1;
        }

        for (auto module : modules) {
            delete module;
        }

        delete g->target;
        g->target = nullptr;

        return errorCount > 0;
    }
}

int Module::LinkAndOutput(std::vector<std::string> linkFiles, OutputType outputType, const char *outFileName) {
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
        if (lIsLlvmBitcode(inputStream))
            m = llvm::parseIRFile(file, err, *g->ctx);
#ifdef ISPC_XE_ENABLED
        else if (lIsSpirVBitcode(inputStream))
            m = translateFromSPIRV(inputStream);
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
    if (outFileName != nullptr) {
        if ((outputType == Bitcode) || (outputType == BitcodeText))
            writeBitcode(llvmLink.get(), outFileName, outputType);
#ifdef ISPC_XE_ENABLED
        else if (outputType == SPIRV)
            writeSPIRV(llvmLink.get(), outFileName);
#endif
        return 0;
    }
    return 1;
}

void Module::clearCPPBuffer() {
    if (bufferCPP)
        bufferCPP.reset();
}
