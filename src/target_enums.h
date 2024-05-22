/*
  Copyright (c) 2019-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file target_enums.h
    @brief Define enums describing target platform.
*/

#pragma once

#include <string>
#include <vector>

namespace ispc {

enum class CallingConv { uninitialized, defaultcall, x86_vectorcall };

enum class TargetOS { windows, linux, custom_linux, freebsd, macos, android, ios, ps4, ps5, web, error };

TargetOS ParseOS(std::string os);
std::string OSToString(TargetOS os);
std::string OSToLowerString(TargetOS os);
TargetOS GetHostOS();

enum class Arch { none, x86, x86_64, arm, aarch64, wasm32, wasm64, xe64, error };

Arch ParseArch(std::string arch);
std::string ArchToString(Arch arch);

enum class ISPCTarget {
    none,
    host,
    sse2_i32x4,
    sse2_i32x8,
    sse41_i8x16,
    sse41_i16x8,
    sse41_i32x4,
    sse41_i32x8,
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
    avx2vnni_i32x4,
    avx2vnni_i32x8,
    avx2vnni_i32x16,
    avx512knl_x16,
    avx512skx_x4,
    avx512skx_x8,
    avx512skx_x16,
    avx512skx_x32,
    avx512skx_x64,
    avx512icl_x4,
    avx512icl_x8,
    avx512icl_x16,
    avx512icl_x32,
    avx512icl_x64,
    avx512spr_x4,
    avx512spr_x8,
    avx512spr_x16,
    avx512spr_x32,
    avx512spr_x64,
    neon_i8x16,
    neon_i16x8,
    neon_i32x4,
    neon_i32x8,
    wasm_i32x4,
    gen9_x8,
    gen9_x16,
    xelp_x8,
    xelp_x16,
    xehpg_x8,
    xehpg_x16,
    xehpc_x16,
    xehpc_x32,
    xelpg_x8,
    xelpg_x16,
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
