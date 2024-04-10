/*
  Copyright (c) 2010-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file builtins.h
    @brief Declarations of functions related to builtins and the
           standard library
*/

#pragma once

#include "builtins-decl.h"
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

void AddBitcodeToModule(const BitcodeLib *lib, llvm::Module *module, SymbolTable *symbolTable = nullptr);

/** Create ISPC symbol for LLVM intrinsics and add it to the given module.

    @param func            llvm::Function for the intrinsic to be added
    @param symbolTable     SymbolTable in which to add symbol definitions
    @return                Symbol created for the LLVM::Function
 */
Symbol *CreateISPCSymbolForLLVMIntrinsic(llvm::Function *func, SymbolTable *symbolTable);

#ifdef ISPC_XE_ENABLED
std::string mangleSPIRVBuiltin(const llvm::Function &func);
#endif

} // namespace ispc
