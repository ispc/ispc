/*
  Copyright (c) 2015-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file ispc_version.h
    @brief defines the ISPC version
*/

#pragma once

#include "../common/version.h"
#include <llvm/Config/llvm-config.h>

#define ISPC_LLVM_VERSION (LLVM_VERSION_MAJOR * 10000 + LLVM_VERSION_MINOR * 100)

#define ISPC_LLVM_17_0 170000
#define ISPC_LLVM_18_1 180100
#define ISPC_LLVM_19_0 190000
#define ISPC_LLVM_20_0 200000
#define ISPC_LLVM_21_0 210000
#define ISPC_LLVM_22_0 220000

#define OLDEST_SUPPORTED_LLVM ISPC_LLVM_17_0
#define LATEST_SUPPORTED_LLVM ISPC_LLVM_22_0

#ifdef __ispc__xstr
#undef __ispc__xstr
#endif
#define __ispc__xstr(s) __ispc__str(s)
#define __ispc__str(s) #s

#define ISPC_LLVM_VERSION_STRING                                                                                       \
    __ispc__xstr(LLVM_VERSION_MAJOR) "." __ispc__xstr(LLVM_VERSION_MINOR) "." __ispc__xstr(LLVM_VERSION_PATCH)

#if ISPC_LLVM_VERSION < OLDEST_SUPPORTED_LLVM || ISPC_LLVM_VERSION > LATEST_SUPPORTED_LLVM
#error "Only LLVM 17.0 - 21.1 and 22.0 development branch are supported"
#endif

#define ISPC_VERSION_STRING                                                                                            \
    "Intel(r) Implicit SPMD Program Compiler (Intel(r) ISPC), " ISPC_VERSION " (build " BUILD_VERSION " @ " BUILD_DATE \
    ", LLVM " ISPC_LLVM_VERSION_STRING ")"
