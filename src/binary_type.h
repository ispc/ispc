/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file dist.h
    @brief Defines function with distribution specific implementations.
*/

#include <llvm/ADT/StringRef.h>

/** Print the binary type: slim or composite. */
void printBinaryType();

/** Initializes distribution specific paths based on main executable abspath. */
void initializeBinaryType(const char *MainExecutableAbsPath);

/** Returns the core.isph file content. */
llvm::StringRef getStdlibISPHRef();

/** Returns the stdlib.isph file content. */
llvm::StringRef getCoreISPHRef();
