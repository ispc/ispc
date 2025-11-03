/*
  Copyright (c) 2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file preprocessor.cpp
    @brief Preprocessor related functions for ISPC.
*/

#include "binary_type.h"
#include "ispc.h"
#include "ispc_version.h"
#include "module.h"
#include "target_capabilities.h"
#include "util.h"
#include "version.h"

#include <ctype.h>
#include <fcntl.h>
#include <memory>
#include <stdio.h>
#include <string>
#include <utility>
#include <vector>

#include <clang/Basic/CharInfo.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticFrontend.h>
#include <clang/Basic/DiagnosticIDs.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/FileEntry.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/FileSystemOptions.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/MacroBuilder.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Frontend/PreprocessorOutputOptions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/Lex/HeaderSearch.h>
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Lex/ModuleLoader.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#ifdef ISPC_HOST_IS_WINDOWS
#include <io.h>
// windows.h defines CALLBACK as __stdcall
// clang/Analysis/CFG.h contains typename with name CALLBACK, which is got screwed up.
// So we include it after clang includes.
#include <windows.h>
#endif

using namespace ispc;

extern int yyparse();
typedef struct yy_buffer_state *YY_BUFFER_STATE;
extern YY_BUFFER_STATE yy_scan_string(const char *);
extern void yy_delete_buffer(YY_BUFFER_STATE);

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

        // Use a direct error reporting to avoid corrupted diagnostic state
        if (inputFile != "-") {
            llvm::errs() << "ISPC: error reading file '" << inputFile << "'\n";
        } else {
            llvm::errs() << "ISPC: error reading stdin: " << errCode.message() << "\n";
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

static void lCreateVirtualHeader(clang::FileManager &fileMgr, clang::SourceManager &srcMgr, llvm::StringRef Filename,
                                 llvm::StringRef Content) {
    std::unique_ptr<llvm::MemoryBuffer> Buffer = llvm::MemoryBuffer::getMemBuffer(Content);
    clang::FileEntryRef FE = fileMgr.getVirtualFileRef(Filename, Buffer->getBufferSize(), 0);
    srcMgr.overrideFileContents(FE, std::move(Buffer));
}

static void lBuilderAppendHeader(clang::MacroBuilder &Builder, std::string &header) {
    Builder.append(llvm::Twine("#include \"") + header + "\"");
}

using StringRefFunc = llvm::StringRef (*)();
static void lAddImplicitInclude(clang::MacroBuilder &Builder, clang::FileManager &fileMgr, clang::SourceManager &srcMgr,
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

    // Define macros for all target capabilities using centralized metadata table
    for (const auto &cm : g_capabilityMetadata) {
        if (g->target->hasCapability(cm.capability)) {
            opts->addMacroDef(cm.macroName);
        }
    }

    // Define mask bits
    std::string ispc_mask_bits = "ISPC_MASK_BITS=" + std::to_string(g->target->getMaskBitCount());
    opts->addMacroDef(ispc_mask_bits);

    if (g->target->is32Bit()) {
        opts->addMacroDef("ISPC_POINTER_SIZE=32");
    } else {
        opts->addMacroDef("ISPC_POINTER_SIZE=64");
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

/** Run the preprocessor on the given file, writing to the output stream.
    Returns the number of diagnostic errors encountered. */
static int lExecPreprocessor(llvm::Module *module, const char *infilename, llvm::raw_string_ostream *ostream,
                             Globals::PreprocessorOutputType preprocessorOutputType) {
    clang::FrontendInputFile inputFile(infilename, clang::InputKind());

    // Create completely isolated diagnostic infrastructure
    // Use a separate error stream for each invocation to avoid shared state
    std::string errorBuffer;
    llvm::raw_string_ostream errorStream(errorBuffer);

    // Create Diagnostic engine with isolated components
#if ISPC_LLVM_VERSION >= ISPC_LLVM_21_0
    clang::DiagnosticOptions diagOptions;
    clang::TextDiagnosticPrinter diagPrinter(errorStream, diagOptions, false);
#else
    llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOptions(new clang::DiagnosticOptions);
    clang::TextDiagnosticPrinter diagPrinter(errorStream, diagOptions.get(), false);
#endif
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagIDs(new clang::DiagnosticIDs);
    clang::DiagnosticsEngine diagEng(diagIDs, diagOptions, &diagPrinter, false);

    diagEng.setSuppressAllDiagnostics(g->ignoreCPPErrors);

    // Create TargetInfo
    const std::shared_ptr<clang::TargetOptions> tgtOpts = std::make_shared<clang::TargetOptions>();
    llvm::Triple triple(module->getTargetTriple());
    if (triple.getTriple().empty()) {
        triple.setTriple(llvm::sys::getDefaultTargetTriple());
    }
    tgtOpts->Triple = triple.getTriple();
#if ISPC_LLVM_VERSION >= ISPC_LLVM_21_0
    clang::TargetInfo *tgtInfo = clang::TargetInfo::CreateTargetInfo(diagEng, *tgtOpts);
#else
    clang::TargetInfo *tgtInfo = clang::TargetInfo::CreateTargetInfo(diagEng, tgtOpts);
#endif

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

    // Output any collected diagnostic messages to stderr
    if (!errorBuffer.empty()) {
        llvm::errs() << errorBuffer;
    }

    // deallocate some objects
    diagIDs.reset();
#if ISPC_LLVM_VERSION < ISPC_LLVM_21_0
    diagOptions.reset();
#endif
    delete tgtInfo;

    // Return preprocessor diagnostic errors after processing
    return static_cast<int>(diagEng.hasErrorOccurred());
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

/** Helper function to initialize the internal CPP buffer. **/
static void lInitCPPBuffer(std::unique_ptr<CPPBuffer> &bufferCPP) {
    // If the CPP stream has been initialized, we have unexpected behavior.
    if (bufferCPP) {
        Assert("CPP stream has already been initialized.");
    }

    // Replace the CPP stream with a newly allocated one.
    bufferCPP.reset(new CPPBuffer{});
}

/** Helper function to parse internal CPP buffer. **/
static void lParseCPPBuffer(std::unique_ptr<CPPBuffer> &bufferCPP) {
    YY_BUFFER_STATE strbuf = yy_scan_string(bufferCPP->str.c_str());
    yyparse();
    yy_delete_buffer(strbuf);
}

/** Helper function to clean internal CPP buffer. **/
static void lClearCPPBuffer(std::unique_ptr<CPPBuffer> &bufferCPP) {
    if (bufferCPP) {
        bufferCPP.reset();
    }
}

int Module::preprocessAndParse() {
    llvm::SmallVector<llvm::StringRef, 6> refs;

    lInitCPPBuffer(bufferCPP);

    const int numErrors = lExecPreprocessor(module, srcFile, bufferCPP->os.get(), g->preprocessorOutputType);
    errorCount += (g->ignoreCPPErrors) ? 0 : numErrors;

    if (g->onlyCPP) {
        return errorCount; // Return early
    }

    lParseCPPBuffer(bufferCPP);
    lClearCPPBuffer(bufferCPP);

    return 0;
}
