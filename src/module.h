/*
  Copyright (c) 2010-2019, Intel Corporation
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

/** @file module.h
    @brief %Declaration of the Module class, which is the ispc-side representation
    of the results of compiling a source file.
 */

#pragma once

#include <string>
#include <vector>

enum class ISPCTarget;

class Expr;
class FunctionType;
class Type;
class Stmt;
class SymbolTable;

namespace llvm {

class raw_string_ostream;

class DIBuilder;
class DICompileUnit;
class Function;
class Module;
class StringRef;

} // namespace llvm

namespace ispc {

enum class Arch;
enum class StorageClass;

struct SourcePos;

class Module final {
  public:
    /** The name of the source file being compiled should be passed as the
        module name. */
    Module(const char *filename);

    Module(Module &&other);

    ~Module();

    /** Compiles the source file passed to the Module constructor, adding
        its global variables and functions to both the llvm::Module and
        SymbolTable.  Returns the number of errors during compilation.  */
    int CompileFile();

    /** Add a named type definition to the module. */
    void AddTypeDef(const std::string &name, const Type *type, SourcePos pos);

    /** Add a new global variable corresponding to the given Symbol to the
        module.  If non-NULL, initExpr gives the initiailizer expression
        for the global's inital value. */
    void AddGlobalVariable(const std::string &name, const Type *type, Expr *initExpr, bool isConst,
                           StorageClass storageClass, SourcePos pos);

    /** Add a declaration of the function defined by the given function
        symbol to the module. */
    void AddFunctionDeclaration(const std::string &name, const FunctionType *ftype, StorageClass sc, bool isInline,
                                bool isNoInline, bool isVectorCall, SourcePos pos);

    /** Adds the function described by the declaration information and the
        provided statements to the module. */
    void AddFunctionDefinition(const std::string &name, const FunctionType *ftype, Stmt *code);

    /** Adds the given type to the set of types that have their definitions
        included in automatically generated header files. */
    void AddExportedTypes(const std::vector<std::pair<const Type *, SourcePos>> &types);

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
#ifdef ISPC_GENX_ENABLED
        ISA,   /** generate GenX ISA file */
        SPIRV, /** generate spir-v file */
#endif
    };

    enum OutputFlags : int {
        NoFlags = 0,
        GeneratePIC = 0x1,
        GenerateFlatDeps = 0x2,        /** Dependencies will be output as a flat list. */
        GenerateMakeRuleForDeps = 0x4, /** Dependencies will be output in a make rule format instead of a flat list. */
        OutputDepsToStdout = 0x8,      /** Dependency information will be output to stdout instead of file. */
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
        @param generatePIC  Indicates whether position-independent code should
                            be generated.
        @param outputType   %Type of output to generate (object files, assembly,
                            LLVM bitcode.)
        @param outFileName  Base name of output filename for object files, etc.
                            If for example the multiple targets "sse2" and "avx"
                            are specified in the "targets" parameter and if this
                            parameter is "foo.o", then we'll generate multiple
                            output files, like "foo.o", "foo_sse2.o", "foo_avx.o".
        @param headerFileName If non-NULL, emit a header file suitable for
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

    /** Accesses the debug info builder.
     * The diBuilder manages generating debugging information.
     *
     * @return May return a pointer to the debug info builder.
     *         In some cases, this function may return a null pointer.
     * */
    llvm::DIBuilder *GetDIBuilder() noexcept;

    llvm::DICompileUnit *GetDICompileUnit() noexcept;

    /** Searches the LLVM module for a function.

       @param name The name of the function to search for.
                  This may also be a string literal.

       @return If the function was found, then a pointer
               to it is returned. Otherwise, nullptr is
               returned instead.
     */
    llvm::Function *GetFunction(const llvm::StringRef &name) const;

    /** Gets a pointer to the LLVM module.

       @return A pointer to the LLVM module.
               This function never returns a
               null pointer.
     */
    llvm::Module *GetLLVMModule() const noexcept;

    SymbolTable &GetSymbolTable() noexcept;

    const SymbolTable &GetSymbolTable() const noexcept;

    bool HasErrors() const noexcept;

    bool IsErrorCountWithin(int limit) const noexcept;

    void IncreaseErrorCount(int n = 1) noexcept;

  private:
    class Impl;
    Impl *self = nullptr;
};

inline Module::OutputFlags &operator|=(Module::OutputFlags &lhs, const __underlying_type(Module::OutputFlags) rhs) {
    return lhs = (Module::OutputFlags)((__underlying_type(Module::OutputFlags))lhs | rhs);
}
inline Module::OutputFlags &operator&=(Module::OutputFlags &lhs, const __underlying_type(Module::OutputFlags) rhs) {
    return lhs = (Module::OutputFlags)((__underlying_type(Module::OutputFlags))lhs & rhs);
}
inline Module::OutputFlags operator|(const Module::OutputFlags lhs, const Module::OutputFlags rhs) {
    return (Module::OutputFlags)((__underlying_type(Module::OutputFlags))lhs | rhs);
}

} // namespace ispc
