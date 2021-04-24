/*
  Copyright (c) 2010-2021, Intel Corporation
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

/** @file builtins.h
    @brief Declarations of functions related to builtins and the
           standard library
*/

#pragma once

#include "ispc.h"

namespace ispc {

/** Adds declarations and definitions of ispc standard library functions
    and types to the given module.

    @param symbolTable     SymbolTable in which to add symbol definitions for
                           stdlib stuff
    @param ctx             llvm::LLVMContext to use for getting types and the
                           like for standard library definitions
    @param module          Module in which to add the declarations/definitions
    @param includeStdlib   Indicates whether the definitions from the stdlib.ispc
                           file should be added to the module.
 */
void DefineStdlib(SymbolTable *symbolTable, llvm::LLVMContext *ctx, llvm::Module *module, bool includeStdlib);

void AddBitcodeToModule(const BitcodeLib *lib, llvm::Module *module, SymbolTable *symbolTable = NULL);

/** Create ISPC symbol for LLVM intrinsics and add it to the given module.

    @param func            llvm::Function for the intrinsic to be added
    @param symbolTable     SymbolTable in which to add symbol definitions
    @return                Symbol created for the LLVM::Function
 */
Symbol *CreateISPCSymbolForLLVMIntrinsic(llvm::Function *func, SymbolTable *symbolTable);

} // namespace ispc
