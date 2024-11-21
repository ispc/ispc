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

/** Add dispatcher to the given module.

    @param module          Module in which to add the declarations/definitions
 */
void LinkDispatcher(llvm::Module *module);

/** Add declarations and definitions for the standard library to the given module.

    @param module          Module in which to add the declarations/definitions
    @param debug_num       Debug number to use for debug output filename
 */
void LinkStandardLibraries(llvm::Module *module, int &debug_num);

/** Dump the given module to a file with the given name.

    @param module          Module to dump
    @param name            Name of the file to which to dump the module
    @param stage           Number to append to the filename to help identify
                           the stage of compilation
 */
void debugDumpModule(llvm::Module *module, std::string name, int stage);

/** Return the suffix to use for target-specific functions.

    @return                Suffix to use for target-specific functions
 */

std::string GetTargetSuffix();

#ifdef ISPC_XE_ENABLED
std::string mangleSPIRVBuiltin(const llvm::Function &func);
#endif

} // namespace ispc
