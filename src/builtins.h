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

// TOOD! rewrite docs
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

void LinkDispatcher(llvm::Module *module);

void LinkCommonBuiltins(SymbolTable *symbolTable, llvm::Module *module);

void LinkTargetBuiltins(SymbolTable *symbolTable, llvm::Module *module);

void LinkStdlib(SymbolTable *symbolTable, llvm::Module *module);

void addPersistentToLLVMUsed(llvm::Module &M);

void removeUnused(llvm::Module *M);

void debugDumpModule(llvm::Module *module, std::string name, int stage);

void AddModuleSymbols(llvm::Module *module, SymbolTable *symbolTable);

void AddBitcodeToModule(llvm::Module *lib, llvm::Module *module);

void AddDeclarationsToModule(llvm::Module *lib, llvm::Module *module);

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
