/*
  Copyright (c) 2019-2021, Intel Corporation
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

/** @file target_enums.h
    @brief Define enums describing target platform.
*/

#pragma once

#include <string>
#include <vector>

namespace ispc {

enum class CallingConv { uninitialized, defaultcall, x86_vectorcall };

enum class TargetOS { windows, linux, custom_linux, freebsd, macos, android, ios, ps4, web, error };

TargetOS ParseOS(std::string os);
std::string OSToString(TargetOS os);
std::string OSToLowerString(TargetOS os);
TargetOS GetHostOS();

enum class Arch { none, x86, x86_64, arm, aarch64, wasm32, genx32, genx64, error };

Arch ParseArch(std::string arch);
std::string ArchToString(Arch arch);

enum class ISPCTarget {
    none,
    host,
    sse2_i32x4,
    sse2_i32x8,
    sse4_i8x16,
    sse4_i16x8,
    sse4_i32x4,
    sse4_i32x8,
    avx1_i32x4,
    avx1_i32x8,
    avx1_i32x16,
    avx1_i64x4,
    avx2_i8x32,
    avx2_i16x16,
    avx2_i32x4,
    avx2_i32x8,
    avx2_i32x16,
    avx2_i64x4,
    avx512knl_i32x16,
    avx512skx_i32x8,
    avx512skx_i32x16,
    avx512skx_i8x64,
    avx512skx_i16x32,
    neon_i8x16,
    neon_i16x8,
    neon_i32x4,
    neon_i32x8,
    wasm_i32x4,
    genx_x8,
    genx_x16,
    error
};

ISPCTarget ParseISPCTarget(std::string target);
std::pair<std::vector<ISPCTarget>, std::string> ParseISPCTargets(const char *target);
std::string ISPCTargetToString(ISPCTarget target);
bool ISPCTargetIsX86(ISPCTarget target);
bool ISPCTargetIsNeon(ISPCTarget target);
bool ISPCTargetIsWasm(ISPCTarget target);
bool ISPCTargetIsGen(ISPCTarget target);
} // namespace ispc
