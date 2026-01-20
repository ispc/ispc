/*
  Copyright (c) 2010-2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file module.h
    @brief %Declaration of the Module class, which is the ispc-side representation
    of the results of compiling a source file.
 */

#pragma once

#include "ast.h"
#include "decl.h"
#include "ispc.h"

#include <algorithm>
#include <memory>
#include <stdio.h>
#include <string>
#include <vector>

#include <clang/Frontend/FrontendOptions.h>
#include <llvm/IR/DebugInfo.h>

#include <llvm/Support/TimeProfiler.h>

#ifdef ISPC_XE_ENABLED
#include <unordered_map>
#endif

namespace llvm {
class raw_string_ostream;
}

namespace ispc {

#ifdef ISPC_XE_ENABLED
// Derived from ocloc_api.h
using invokePtr = int (*)(unsigned, const char **, const uint32_t, const uint8_t **, const uint64_t *, const char **,
                          const uint32_t, const uint8_t **, const uint64_t *, const char **, uint32_t *, uint8_t ***,
                          uint64_t **, char ***);
using freeOutputPtr = int (*)(uint32_t *, uint8_t ***, uint64_t **, char ***);
#endif

// Definition and member object capturing preprocessing stream during Module lifetime.
struct CPPBuffer {
    CPPBuffer() : str{}, os{std::make_unique<llvm::raw_string_ostream>(str)} {}
    ~CPPBuffer() = default;
    std::string str;
    std::unique_ptr<llvm::raw_string_ostream> os;
};

/**
 * @class Module
 * Main container for all data related to a compiled ISPC translation unit.
 *
 * The Module class manages the compilation process for a single ISPC source file.
 * It contains the AST, symbol table, and LLVM module produced during compilation,
 * and provides methods for all stages of the compilation process including preprocessing,
 * parsing, optimization, and output generation.
 */
class Module {
  public:
    /** The name of the source file being compiled should be passed as the
        module name. */
    Module(const char *filename);

    ~Module();

    // We don't copy Module objects at the moment. If we will then proper
    // implementations are needed considering the ownership of heap-allocated
    // fields like symbolTable.
    Module(const Module &) = delete;
    Module &operator=(const Module &) = delete;

    /** Compiles the source file passed to the Module constructor, adding
        its global variables and functions to both the llvm::Module and
        SymbolTable.  Returns the number of errors during compilation.  */
    int CompileFile();

    /** Add a named type definition to the module. */
    void AddTypeDef(const std::string &name, const Type *type, SourcePos pos);

    /** Add a new global variable corresponding to the decl. */
    void AddGlobalVariable(Declarator *decl, bool isConst, bool isConstexpr);

    /** Record a deferred constexpr default parameter for later evaluation. */
    void AddDeferredConstexprParam(FunctionType *funcType, int index, Expr *expr, SourcePos pos,
                                   const std::string &paramName);

    /** Add a declaration of the function defined by the given function
        symbol to the module. */
    void AddFunctionDeclaration(const std::string &name, const FunctionType *ftype, StorageClass sc, Declarator *decl,
                                bool isInline, bool isNoInline, bool isVectorCall, bool isRegCall, SourcePos pos);

    /** Adds the function described by the declaration information and the
        provided statements to the module. */
    void AddFunctionDefinition(const std::string &name, const FunctionType *ftype, Stmt *code);

    /** Add a declaration of the function template defined by the given function
        symbol to the module. */
    void AddFunctionTemplateDeclaration(const TemplateParms *templateParmList, const std::string &name,
                                        const FunctionType *ftype, StorageClass sc, bool isInline, bool isNoInline,
                                        SourcePos pos);

    /** Add the function described by the declaration information and the
        provided statements to the module. */
    void AddFunctionTemplateDefinition(const TemplateParms *templateParmList, const std::string &name,
                                       const FunctionType *ftype, Stmt *code);

    void AddFunctionTemplateInstantiation(const std::string &name, const TemplateArgs &tArgs, const FunctionType *ftype,
                                          StorageClass sc, bool isInline, bool isNoInline, SourcePos pos);

    void AddFunctionTemplateSpecializationDeclaration(const std::string &name, const FunctionType *ftype,
                                                      const TemplateArgs &tArgs, StorageClass sc, bool isInline,
                                                      bool isNoInline, SourcePos pos);

    void AddFunctionTemplateSpecializationDefinition(const std::string &name, const FunctionType *ftype,
                                                     const TemplateArgs &tArgs, SourcePos pos, Stmt *code);

    /** Adds the given type to the set of types that have their definitions
        included in automatically generated header files. */
    void AddExportedTypes(const std::vector<std::pair<const Type *, SourcePos>> &types);

    /** Verify LLVM intrinsic called from ISPC source code is valid and return
        function symbol for it. */
    Symbol *AddLLVMIntrinsicDecl(const std::string &name, ExprList *args, SourcePos po);

    /** Returns pointer to FunctionTemplate based on template name and template argument types provided. Also makes
       template argument types normalization, i.e apply "varying type default":
       template <typename T> void foo(T t);
       foo<int>(1); // T is assumed to be "varying int" here.
    */
    FunctionTemplate *MatchFunctionTemplate(const std::string &name, const FunctionType *ftype, TemplateArgs &normTypes,
                                            SourcePos pos);

    /** After a source file has been compiled, output can be generated in a
        number of different formats. */
    enum OutputType {
        Asm = 0,         /** Generate text assembly language output */
        Bitcode,         /** Generate LLVM IR bitcode output */
        BitcodeText,     /** Generate LLVM IR Text output */
        Object,          /** Generate a native object file */
        Header,          /** Generate a C/C++ header file with
                             declarations of 'export'ed functions, global
                             variables, and the types used by them. */
        NanobindWrapper, /** Generate a C++ wrapper file for nanobind */
        Deps,            /** generate dependencies */
        DevStub,         /** generate device-side offload stubs */
        HostStub,        /** generate host-side offload stubs */
        CPPStub,         /** generate preprocessed stubs (-E/-dD/-dM mode) */
#ifdef ISPC_XE_ENABLED
        ZEBIN, /** generate L0 binary file */
        SPIRV, /** generate spir-v file */
#endif
    };

    // Define a mapping from OutputType to expected suffixes and file type descriptions
    struct OutputTypeInfo {
        const char *fileType;
        std::vector<std::string> validSuffixes;

        // Check if the provided suffix is valid for this fileType by
        // case-insensitive comparison with valid suffixes that are stored in
        // validSuffixes vector
        bool isSuffixValid(const std::string &suffix) const {
            // dependency suffixes are empty
            if (validSuffixes.empty()) {
                return true;
            }
            return std::find_if(validSuffixes.begin(), validSuffixes.end(), [&suffix](const std::string &valid) {
                       return std::equal(suffix.begin(), suffix.end(), valid.begin(), valid.end(),
                                         [](char a, char b) { return std::tolower(a) == std::tolower(b); });
                   }) != validSuffixes.end();
        }
    };

    /**
     * @class OutputFlags
     * Flags that control output file generation
     */
    class OutputFlags {
      public:
        OutputFlags()
            : picLevel(PICLevel::Default), flatDeps(false), makeRuleDeps(false), depsToStdout(false),
              mcModel(MCModel::Default) {}
        OutputFlags(OutputFlags &o)
            : picLevel(o.picLevel), flatDeps(o.flatDeps), makeRuleDeps(o.makeRuleDeps), depsToStdout(o.depsToStdout),
              mcModel(o.mcModel) {}

        OutputFlags &operator=(const OutputFlags &o) {
            picLevel = o.picLevel;
            flatDeps = o.flatDeps;
            makeRuleDeps = o.makeRuleDeps;
            depsToStdout = o.depsToStdout;
            mcModel = o.mcModel;
            return *this;
        };

        void setPICLevel(PICLevel v = PICLevel::Default) { picLevel = v; }
        PICLevel getPICLevel() const { return picLevel; }
        bool isPIC() const { return picLevel != PICLevel::Default; }
        void setFlatDeps(bool v = true) { flatDeps = v; }
        bool isFlatDeps() const { return flatDeps; }
        void setMakeRuleDeps(bool v = true) { makeRuleDeps = v; }
        bool isMakeRuleDeps() const { return makeRuleDeps; }
        void setDepsToStdout(bool v = true) { depsToStdout = v; }
        bool isDepsToStdout() const { return depsToStdout; }
        void setMCModel(MCModel m) { mcModel = m; }
        MCModel getMCModel() const { return mcModel; }

      private:
        // --pic --PIC
        PICLevel picLevel;
        // -MMM
        bool flatDeps;
        // -M
        bool makeRuleDeps;
        // deps output to stdout
        bool depsToStdout;
        // --mcmodel value
        MCModel mcModel;
    };

    /**
     * @struct Output
     * Encapsulates all output-related information for the compiler
     *
     * This struct contains the output type (object file, assembly, bitcode, etc.),
     * output flags, and various file paths for different output artifacts that can
     * be generated during compilation.
     *
     * The compiler can generate multiple types of output files in a single compilation:
     * - Main output (object file, assembly, bitcode, etc.)
     * - Header file with declarations for C/C++ interoperability
     * - Dependency information files (makefile)
     * - Host and device stub files for offload models
     */
    struct Output {
        Output() {}

        /**
         * Constructor that sets all output parameters
         *
         * @param outputType        Type of output to generate
         * @param outputFlags       Flags controlling output generation
         * @param outFileName       Main output file name
         * @param headerFileName    Header file name
         * @param nbWrapFileName    Nanobind wrapper file name
         * @param depsFileName      Dependencies file name
         * @param hostStubFileName  Host stub file name
         * @param devStubFileName   Device stub file name
         * @param depsTargetName     Dependencies target name
         */
        Output(OutputType outputType, OutputFlags outputFlags, const char *outFileName, const char *headerFileName,
               const char *nbWrapFileName, const char *depsFileName, const char *hostStubFileName,
               const char *devStubFileName, const char *depsTargetName)
            : type(outputType), flags(outputFlags), depsTarget(depsTargetName ? depsTargetName : ""),
              out(outFileName ? outFileName : ""), header(headerFileName ? headerFileName : ""),
              nbWrap(nbWrapFileName ? nbWrapFileName : ""), deps(depsFileName ? depsFileName : ""),
              hostStub(hostStubFileName ? hostStubFileName : ""), devStub(devStubFileName ? devStubFileName : "") {}

        OutputType type{};
        OutputFlags flags{};

        std::string depsTarget{};

        // Output file names
        std::string out{};      /**< Main output file name */
        std::string header{};   /**< Header file name */
        std::string nbWrap{};   /**< Nanobind wrapper file name */
        std::string deps{};     /**< Dependencies file name */
        std::string hostStub{}; /**< Host stub file name */
        std::string devStub{};  /**< Device stub file name */

        /**
         * Get the target name for dependencies
         *
         * @param srcFile Source file name
         * @return The target name for dependencies
         */
        std::string DepsTargetName(const char *srcFile) const;

        /**
         * Get the output file name for a specific target
         *
         * Base name of output filename for object files, etc. If it operates
         * in multi-target mode, it will append the target name to the base
         * name, e.g. "foo.o" -> "foo_sse2.o".
         *
         * @param target Target to generate filename for
         * @return The output file name for the target
         */
        std::string OutFileNameTarget(Target *target) const;

        /**
         * Get the header file name for a specific target
         *
         * @param target Target to generate filename for
         * @return The header file name for the target
         */
        std::string HeaderFileNameTarget(Target *target) const;
    };

    /**
     * @struct DispatchHeaderInfo
     * Information for generating dispatch headers
     */
    struct DispatchHeaderInfo {
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

        bool initialize(std::string headerFileName);

        void closeFile();

        ~DispatchHeaderInfo() { closeFile(); }
    };

    /**
     * @enum CompilationMode
     * Defines whether the module is being compiled alone or as part of a multi-target compilation
     */
    enum class CompilationMode { Single, Multiple };

    /**
     * Constructor with output and compilation mode parameters
     *
     * @param filename Source file name
     * @param output Output settings
     * @param mode Compilation mode
     */
    Module(const char *filename, Output &output, CompilationMode mode);

    /**
     * Factory method to create a Module instance
     *
     * @param srcFile Source file name
     * @param output Output settings
     * @param mode Compilation mode (default: Single)
     * @return A unique_ptr to a new Module instance
     */
    static std::unique_ptr<Module> Create(const char *srcFile, Output &output,
                                          CompilationMode mode = CompilationMode::Single);

    /** Compile the given source file, generating assembly, object file, or
        LLVM bitcode output, as well as (optionally) a header file with
        declarations of functions and types used in the ispc/application
        interface.
        @param srcFile      Pathname to ispc source file to compile
        @param arch         %Target architecture (e.g. "x86-64")
        @param cpu          %Target CPU (e.g. "core-i7")
        @param targets      %Target ISAs; this parameter may give a single target
                            ISA, or may give a comma-separated list of them in
                            case we are compiling to multiple ISAs.
        @param output       Output settings
        @return             Number of errors encountered when compiling
                            srcFile.
     */
    static int CompileAndOutput(const char *srcFile, Arch arch, const char *cpu, std::vector<ISPCTarget> &targets,
                                Output &output);

    static std::unique_ptr<llvm::Module> CompileToLLVMModule(const char *srcFile, Arch arch, const char *cpu,
                                                             std::vector<ISPCTarget> &targets);

    static int LinkAndOutput(std::vector<std::string> linkFiles, OutputType outputType, std::string outFileName);

    const char *RegisterDependency(const std::string &fileName);

    /** Total number of errors encountered during compilation. */
    int errorCount{0};

    /** Symbol table to hold symbols visible in the current scope during
        compilation. */
    SymbolTable *symbolTable{nullptr};

    /** llvm Module object into which globals and functions are added. */
    llvm::Module *module{nullptr};

    /** The diBuilder manages generating debugging information */
    llvm::DIBuilder *diBuilder{nullptr};

    llvm::DICompileUnit *diCompileUnit{nullptr};

    /** StructType cache.  This needs to be in the context of Module, so it's reset for
        any new Module in multi-target compilation.

        We maintain a map from struct names to LLVM struct types so that we can
        uniquely get the llvm::StructType * for a given ispc struct type.  Note
        that we need to mangle the name a bit so that we can e.g. differentiate
        between the uniform and varying variants of a given struct type.  This
        is handled by lMangleStructName() below. */
    std::map<std::string, llvm::StructType *> structTypeMap;

  private:
    struct DeferredConstexprGlobal {
        Symbol *sym;
        Expr *initExpr;
        const Type *type;
        SourcePos pos;
        bool isConst;
        bool isConstexpr;
    };

    struct DeferredConstexprParam {
        FunctionType *funcType;
        int index;
        Expr *expr;
        SourcePos pos;
        std::string paramName;
    };

    const char *srcFile{nullptr};
    AST *ast{nullptr};

    Output output{};
    CompilationMode m_compilationMode{};

    std::unique_ptr<CPPBuffer> bufferCPP{nullptr};

    std::vector<std::pair<const Type *, SourcePos>> exportedTypes;

    const std::vector<OutputTypeInfo> outputTypeInfos = {
        /* Asm         */ {"assembly", {"s"}},
        /* Bitcode     */ {"LLVM bitcode", {"bc"}},
        /* BitcodeText */ {"LLVM assembly", {"ll"}},
        /* Object      */ {"object", {"o", "obj"}},
        /* Header      */ {"header", {"h", "hh", "hpp"}},
        /* Nanobind    */ {"nanobind wrapper", {"cpp", "cxx", "cc"}},
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

    /*! list of files encountered by the parser. this allows emitting of
        the module file's dependencies via the -MMM option */
    std::set<std::string> registeredDependencies;

    /* This set is used to store strings that is referenced in yylloc (SourcePos)
       in lexer once and not to lost memory via just strduping them. */
    std::set<std::string> pseudoDependencies;

    std::vector<DeferredConstexprGlobal> deferredConstexprGlobals;
    std::vector<DeferredConstexprParam> deferredConstexprParams;

    void ResolveDeferredConstexpr();

    /**
     * Compiles the module for a single target architecture.
     *
     * @param arch       The target architecture
     * @param cpu        The target CPU
     * @param target     The specific ISA target
     *
     * @return The number of errors encountered during compilation
     */
    int CompileSingleTarget(Arch arch, const char *cpu, ISPCTarget target);

    /**
     * Compiles the given source file for multiple target ISAs and creates a dispatch module.
     *
     * @param srcFile  Path to the source file to compile
     * @param arch     Target architecture
     * @param cpu      CPU specification
     * @param targets  Vector of ISA targets to compile for
     * @param output   Output configuration including output type and filenames
     *
     * @return The number of errors encountered during compilation
     */
    static int CompileMultipleTargets(const char *srcFile, Arch arch, const char *cpu, std::vector<ISPCTarget> &targets,
                                      Output &output);

    /**
     * Creates and outputs a dispatch module for multi-target compilation.
     *
     * @param srcFile     Source file path (for error reporting)
     * @param targets     Vector of ISA targets that were compiled
     * @param modules     Vector of Module instances for each target
     * @param targetsPtrs Vector of Target instances for each target
     * @param output      Output configuration for the dispatch module
     *
     * @return The number of errors encountered during dispatch generation
     */
    static int GenerateDispatch(const char *srcFile, std::vector<ISPCTarget> &targets,
                                std::vector<std::unique_ptr<Module>> &modules,
                                std::vector<std::unique_ptr<Target>> &targetsPtrs, Output &output);

    /**
     * Writes the dispatch module to the appropriate output files.
     *
     * @param dispatchModule The LLVM module containing the dispatch code
     * @param output         Output configuration specifying file types and paths
     *
     * @return Zero on success, non-zero on error
     */
    static int WriteDispatchOutputFiles(llvm::Module *dispatchModule, Output &output);

    static bool writeBitcode(llvm::Module *module, std::string outFileName, OutputType outputType);

    /**
     * Writes all output files for the current module.
     *
     * @return Zero on success, non-zero if any file writing operation failed
     */
    int WriteOutputFiles();

    /** Check if the given output type is valid for the specified file name
      suffix. If not, print a warning message. Correct suffixes are defined in
      outputTypeInfos. */
    void reportInvalidSuffixWarning(std::string filename, OutputType outputType);

    /** Write the corresponding output type to the given file.  Returns
        true on success, false if there has been an error.  The given
        filename may be nullptr, indicating that output should go to standard
        output. */
    bool writeOutput();

    bool writeHeader();
    void writeHeader(FILE *f);
    bool writeNanobindWrapper();
    bool writeDispatchHeader(DispatchHeaderInfo *DHI);

    /**
     * Generates a dependency file in make-compatible format.
     *
     * @param customOutput Output configuration specifying dependency file path
     * and format
     *
     * @return True on success, false if file creation or writing failed
     */
    bool writeDeps(Output &customOutput);
    bool writeDevStub();
    bool writeHostStub();
    bool writeCPPStub();
    bool writeObjectFileOrAssembly(llvm::Module *module, Output &customOutput);
#ifdef ISPC_XE_ENABLED
    static std::unique_ptr<llvm::Module> translateFromSPIRV(std::ifstream &outString);
    static bool translateToSPIRV(llvm::Module *module, std::stringstream &outString);
    static bool writeSPIRV(llvm::Module *module, std::string outFileName);
    bool writeZEBin();
#endif

    int preprocessAndParse();
    int parse();
};

} // namespace ispc
