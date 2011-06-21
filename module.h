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

class Module {
public:
    /** The name of the source file being compiled should be passed as the
        module name. */
    Module(const char *filename);

    /** Compiles the source file passed to the Module constructor, adding
        its global variables and functions to both the llvm::Module and
        SymbolTable.  Returns the number of errors during compilation.  */
    int CompileFile();

    /** Adds the global variable described by the declaration information to
        the module. */
    void AddGlobal(DeclSpecs *ds, Declarator *decl);

    /** Adds the function described by the declaration information and the
        provided statements to the module. */
    void AddFunction(DeclSpecs *ds, Declarator *decl, Stmt *code);

    /** After a source file has been compiled, output can be generated in a
        number of different formats. */
    enum OutputType { Asm,      /** Generate text assembly language output */
                      Bitcode,  /** Generate LLVM IR bitcode output */
                      Object,   /** Generate a native object file */
                      Header    /** Generate a C/C++ header file with 
                                    declarations of 'export'ed functions, global
                                    variables, and the types used by them. */
    };

    /** Write the corresponding output type to the given file.  Returns
        true on success, false if there has been an error.  The given
        filename may be NULL, indicating that output should go to standard
        output. */
    bool WriteOutput(OutputType ot, const char *filename);

    /** Total number of errors encountered during compilation. */
    int errorCount;

    /** Symbol table to hold symbols visible in the current scope during
        compilation. */
    SymbolTable *symbolTable;

    /** llvm Module object into which globals and functions are added. */
    llvm::Module *module; 

#ifndef LLVM_2_8
    /** The diBuilder manages generating debugging information (only
        supported in LLVM 2.9 and beyond...) */
    llvm::DIBuilder *diBuilder;
#endif

    GatherBuffer *gatherBuffer;

private:
    const char *filename;

    /** This member records the global variables that have been defined
        with 'extern' linkage, so that it's easy to include their
        declarations in generated header files.

        @todo FIXME: it would be nice to eliminate this and then query the
        symbol table or the llvm Module for them when/if we need them.
     */
    std::vector<Symbol *> externGlobals;

    bool writeHeader(const char *filename);
    bool writeObjectFileOrAssembly(OutputType outputType, const char *filename);
};

#endif // ISPC_MODULE_H
