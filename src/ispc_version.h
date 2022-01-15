/*
  Copyright (c) 2015-2022, Intel Corporation
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

/** @file ispc_version.h
    @brief defines the ISPC version
*/

#pragma once

#define ISPC_VERSION_MAJOR 1
#define ISPC_VERSION_MINOR 17
#define ISPC_VERSION "1.17.0"
#include <llvm/Config/llvm-config.h>

#define ISPC_LLVM_VERSION (LLVM_VERSION_MAJOR * 10000 + LLVM_VERSION_MINOR * 100)

#define ISPC_LLVM_10_0 100000
#define ISPC_LLVM_11_0 110000
#define ISPC_LLVM_11_1 110100
#define ISPC_LLVM_12_0 120000
#define ISPC_LLVM_13_0 130000
#define ISPC_LLVM_14_0 140000

#define OLDEST_SUPPORTED_LLVM ISPC_LLVM_10_0
#define LATEST_SUPPORTED_LLVM ISPC_LLVM_14_0

#ifdef __ispc__xstr
#undef __ispc__xstr
#endif
#define __ispc__xstr(s) __ispc__str(s)
#define __ispc__str(s) #s

#define ISPC_LLVM_VERSION_STRING                                                                                       \
    __ispc__xstr(LLVM_VERSION_MAJOR) "." __ispc__xstr(LLVM_VERSION_MINOR) "." __ispc__xstr(LLVM_VERSION_PATCH)

#if ISPC_LLVM_VERSION < OLDEST_SUPPORTED_LLVM || ISPC_LLVM_VERSION > LATEST_SUPPORTED_LLVM
#error "Unhandled LLVM version"
#endif

#define ISPC_VERSION_STRING                                                                                            \
    "Intel(r) Implicit SPMD Program Compiler (Intel(r) ISPC), " ISPC_VERSION " (build " BUILD_VERSION " @ " BUILD_DATE \
    ", LLVM " ISPC_LLVM_VERSION_STRING ")"
