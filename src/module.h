/*
  Copyright (c) 2010-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file module.h
    @brief %Declaration of the Module class, which is the ispc-side representation
    of the results of compiling a source file.
 */

#pragma once

#include "ast.h"
#include "ispc.h"

#include <llvm/IR/DebugInfo.h>

#include <llvm/Support/TimeProfiler.h>

#ifdef ISPC_XE_ENABLED
#include <unordered_map>
#endif

namespace llvm {
class raw_string_ostream;
}

namespace ispc {

struct DispatchHeaderInfo;

#ifdef ISPC_XE_ENABLED
// Derived from ocloc_api.h
using invokePtr = int (*)(unsigned, const char **, const uint32_t, const uint8_t **, const uint64_t *, const char **,
                          const uint32_t, const uint8_t **, const uint64_t *, const char **, uint32_t *, uint8_t ***,
                          uint64_t **, char ***);
using freeOutputPtr = int (*)(uint32_t *, uint8_t ***, uint64_t **, char ***);
#endif

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

    /** Add a new global variable corresponding to the given Symbol to the
        module.  If non-nullptr, initExpr gives the initiailizer expression
        for the global's inital value. */
    void AddGlobalVariable(const std::string &name, const Type *type, Expr *initExpr, bool isConst,
                           StorageClass storageClass, SourcePos pos);

    /** Add a declaration of the function defined by the given function
        symbol to the module. */
    void AddFunctionDeclaration(const std::string &name, const FunctionType *ftype, StorageClass sc, bool isInline,
                                bool isNoInline, bool isVectorCall, bool isRegCall, SourcePos pos);

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
        Asm,         /** Generate text assembly language output */
        Bitcode,     /** Generate LLVM IR bitcode output */
        BitcodeText, /** Generate LLVM IR Text output */
        Object,      /** Generate a native object file */
        Header,      /** Generate a C/C++ header file with
                         declarations of 'export'ed functions, global
                         variables, and the types used by them. */
        Deps,        /** generate dependencies */
        DevStub,     /** generate device-side offload stubs */
        HostStub,    /** generate host-side offload stubs */
        CPPStub,     /** generate preprocessed stubs (-E mode) */
#ifdef ISPC_XE_ENABLED
        ZEBIN, /** generate L0 binary file */
        SPIRV, /** generate spir-v file */
#endif
    };

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
        @param OutputFlags  A set of flags for output generation.
        @param outputType   %Type of output to generate (object files, assembly,
                            LLVM bitcode.)
        @param outFileName  Base name of output filename for object files, etc.
                            If for example the multiple targets "sse2" and "avx"
                            are specified in the "targets" parameter and if this
                            parameter is "foo.o", then we'll generate multiple
                            output files, like "foo.o", "foo_sse2.o", "foo_avx.o".
        @param headerFileName If non-nullptr, emit a header file suitable for
                              inclusion from C/C++ code with declarations of
                              types and functions exported from the given ispc
                              source file.
        @return             Number of errors encountered when compiling
                            srcFile.
     */
    static int CompileAndOutput(const char *srcFile, Arch arch, const char *cpu, std::vector<ISPCTarget> targets,
                                OutputFlags outputFlags, OutputType outputType, const char *outFileName,
                                const char *headerFileName, const char *depsFileName, const char *depsTargetName,
                                const char *hostStubFileName, const char *devStubFileName);
    static int LinkAndOutput(std::vector<std::string> linkFiles, OutputType outputType, const char *outFileName);

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
    const char *filename{nullptr};
    AST *ast{nullptr};

    // Definition and member object capturing preprocessing stream during Module lifetime.
    struct CPPBuffer {
        CPPBuffer() : str{}, os{std::make_unique<llvm::raw_string_ostream>(str)} {}
        ~CPPBuffer() = default;
        std::string str;
        std::unique_ptr<llvm::raw_string_ostream> os;
    };

    std::unique_ptr<CPPBuffer> bufferCPP{nullptr};

    std::vector<std::pair<const Type *, SourcePos>> exportedTypes;

    /** Write the corresponding output type to the given file.  Returns
        true on success, false if there has been an error.  The given
        filename may be nullptr, indicating that output should go to standard
        output. */
    bool writeOutput(OutputType ot, OutputFlags flags, const char *filename, const char *depTargetFileName = nullptr,
                     const char *sourceFileName = nullptr, DispatchHeaderInfo *DHI = 0);
    bool writeHeader(const char *filename);
    bool writeDispatchHeader(DispatchHeaderInfo *DHI);
    bool writeDeps(const char *filename, bool generateMakeRule, const char *targetName = nullptr,
                   const char *srcFilename = nullptr);
    bool writeDevStub(const char *filename);
    bool writeHostStub(const char *filename);
    bool writeCPPStub(const char *outFileName);
    bool writeObjectFileOrAssembly(OutputType outputType, const char *filename);
    static bool writeCPPStub(Module *module, const char *outFileName);
    static bool writeObjectFileOrAssembly(llvm::TargetMachine *targetMachine, llvm::Module *module,
                                          OutputType outputType, const char *outFileName);
    static bool writeBitcode(llvm::Module *module, const char *outFileName, OutputType outputType);
#ifdef ISPC_XE_ENABLED
    static std::unique_ptr<llvm::Module> translateFromSPIRV(std::ifstream &outString);
    static bool translateToSPIRV(llvm::Module *module, std::stringstream &outString);
    static bool writeSPIRV(llvm::Module *module, const char *outFileName);
    static bool writeZEBin(llvm::Module *module, const char *outFileName);
#endif

    /** Run the preprocessor on the given file, writing to the output stream.
        Returns the number of diagnostic errors encountered. */
    int execPreprocessor(const char *infilename, llvm::raw_string_ostream *ostream) const;

    /** Helper function to clean internal CPP buffer. **/
    void clearCPPBuffer();
};

} // namespace ispc
