/*
  Copyright (c) 2010-2011, Intel Corporation
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
    @brief Declaration of the Module class, which is the ispc-side representation
    of the results of compiling a source file.
 */

#ifndef ISPC_MODULE_H
#define ISPC_MODULE_H 1

#include "ispc.h"
#include "ast.h"

namespace llvm
{
    class raw_string_ostream;
}

class Module {
public:
    /** The name of the source file being compiled should be passed as the
        module name. */
    Module(const char *filename);

    /** Compiles the source file passed to the Module constructor, adding
        its global variables and functions to both the llvm::Module and
        SymbolTable.  Returns the number of errors during compilation.  */
    int CompileFile();

    /** Add a named type definition to the module. */
    void AddTypeDef(Symbol *sym);

    /** Add a new global variable corresponding to the given Symbol to the
        module.  If non-NULL, initExpr gives the initiailizer expression
        for the global's inital value. */ 
    void AddGlobalVariable(Symbol *sym, Expr *initExpr, bool isConst);

    /** Add a declaration of the function defined by the given function
        symbol to the module. */
    void AddFunctionDeclaration(Symbol *funSym, bool isInline);

    /** Adds the function described by the declaration information and the
        provided statements to the module. */
    void AddFunctionDefinition(Symbol *sym, const std::vector<Symbol *> &args,
                               Stmt *code);

    /** After a source file has been compiled, output can be generated in a
        number of different formats. */
    enum OutputType { Asm,      /** Generate text assembly language output */
                      Bitcode,  /** Generate LLVM IR bitcode output */
                      Object,   /** Generate a native object file */
#ifndef LLVM_2_9
                      CXX,      /** Generate a C++ file */
#endif // !LLVM_2_9
                      Header    /** Generate a C/C++ header file with 
                                    declarations of 'export'ed functions, global
                                    variables, and the types used by them. */
    };

    /** Compile the given source file, generating assembly, object file, or
        LLVM bitcode output, as well as (optionally) a header file with
        declarations of functions and types used in the ispc/application
        interface.
        @param srcFile      Pathname to ispc source file to compile
        @param arch         Target architecture (e.g. "x86-64")
        @param cpu          Target CPU (e.g. "core-i7")
        @param targets      Target ISAs; this parameter may give a single target
                            ISA, or may give a comma-separated list of them in
                            case we are compiling to multiple ISAs.
        @param generatePIC  Indicates whether position-independent code should
                            be generated.
        @param outputType   Type of output to generate (object files, assembly,
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
        @param includeFileName If non-NULL, gives the filename for the C++ 
                               backend to emit in an #include statement to
                               get definitions of the builtins for the generic
                               target.
        @return             Number of errors encountered when compiling
                            srcFile.
     */
    static int CompileAndOutput(const char *srcFile, const char *arch, 
                                const char *cpu, const char *targets, 
                                bool generatePIC, OutputType outputType, 
                                const char *outFileName, 
                                const char *headerFileName, 
                                const char *includeFileName);

    /** Total number of errors encountered during compilation. */
    int errorCount;

    /** Symbol table to hold symbols visible in the current scope during
        compilation. */
    SymbolTable *symbolTable;

    /** llvm Module object into which globals and functions are added. */
    llvm::Module *module; 

    /** The diBuilder manages generating debugging information */
    llvm::DIBuilder *diBuilder;

private:
    const char *filename;
    AST *ast;

    /** Write the corresponding output type to the given file.  Returns
        true on success, false if there has been an error.  The given
        filename may be NULL, indicating that output should go to standard
        output. */
    bool writeOutput(OutputType ot, const char *filename,
                     const char *includeFileName = NULL);
    bool writeHeader(const char *filename);
    bool writeObjectFileOrAssembly(OutputType outputType, const char *filename);
    static bool writeObjectFileOrAssembly(llvm::TargetMachine *targetMachine,
                                          llvm::Module *module, OutputType outputType, 
                                          const char *outFileName);
    static bool writeBitcode(llvm::Module *module, const char *outFileName);

    void execPreprocessor(const char *infilename, llvm::raw_string_ostream* ostream) const;
};

#endif // ISPC_MODULE_H
