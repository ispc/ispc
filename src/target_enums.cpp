/*
  Copyright (c) 2019-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file target_enums.cpp
    @brief Define enums describing target platform.
*/

#include "target_enums.h"

#include "ispc.h"
#include "util.h"

#include <cstring>

namespace ispc {

TargetOS operator++(TargetOS &os, int dummy) {
    using underlying = std::underlying_type_t<TargetOS>;
    static_assert(static_cast<underlying>(TargetOS::linux) == static_cast<underlying>(TargetOS::windows) + 1,
                  "Enum TargetOS is not sequential");
    static_assert(static_cast<underlying>(TargetOS::custom_linux) == static_cast<underlying>(TargetOS::linux) + 1,
                  "Enum TargetOS is not sequential");
    static_assert(static_cast<underlying>(TargetOS::freebsd) == static_cast<underlying>(TargetOS::custom_linux) + 1,
                  "Enum TargetOS is not sequential");
    static_assert(static_cast<underlying>(TargetOS::macos) == static_cast<underlying>(TargetOS::freebsd) + 1,
                  "Enum TargetOS is not sequential");
    static_assert(static_cast<underlying>(TargetOS::android) == static_cast<underlying>(TargetOS::macos) + 1,
                  "Enum TargetOS is not sequential");
    static_assert(static_cast<underlying>(TargetOS::ios) == static_cast<underlying>(TargetOS::android) + 1,
                  "Enum TargetOS is not sequential");
    static_assert(static_cast<underlying>(TargetOS::ps4) == static_cast<underlying>(TargetOS::ios) + 1,
                  "Enum TargetOS is not sequential");
    static_assert(static_cast<underlying>(TargetOS::ps5) == static_cast<underlying>(TargetOS::ps4) + 1,
                  "Enum TargetOS is not sequential");
    static_assert(static_cast<underlying>(TargetOS::web) == static_cast<underlying>(TargetOS::ps5) + 1,
                  "Enum TargetOS is not sequential");
    static_assert(static_cast<underlying>(TargetOS::error) == static_cast<underlying>(TargetOS::web) + 1,
                  "Enum TargetOS is not sequential");
    return os = static_cast<TargetOS>(static_cast<underlying>(os) + 1);
}

Arch operator++(Arch &arch, int dummy) {
    using underlying = std::underlying_type_t<Arch>;
    static_assert(static_cast<underlying>(Arch::x86) == static_cast<underlying>(Arch::none) + 1,
                  "Enum Arch is not sequential");
    static_assert(static_cast<underlying>(Arch::x86_64) == static_cast<underlying>(Arch::x86) + 1,
                  "Enum Arch is not sequential");
    static_assert(static_cast<underlying>(Arch::arm) == static_cast<underlying>(Arch::x86_64) + 1,
                  "Enum Arch is not sequential");
    static_assert(static_cast<underlying>(Arch::aarch64) == static_cast<underlying>(Arch::arm) + 1,
                  "Enum Arch is not sequential");
    static_assert(static_cast<underlying>(Arch::riscv64) == static_cast<underlying>(Arch::aarch64) + 1,
                  "Enum Arch is not sequential");
    static_assert(static_cast<underlying>(Arch::wasm32) == static_cast<underlying>(Arch::riscv64) + 1,
                  "Enum Arch is not sequential");
    static_assert(static_cast<underlying>(Arch::wasm64) == static_cast<underlying>(Arch::wasm32) + 1,
                  "Enum Arch is not sequential");
    static_assert(static_cast<underlying>(Arch::xe64) == static_cast<underlying>(Arch::wasm64) + 1,
                  "Enum Arch is not sequential");
    static_assert(static_cast<underlying>(Arch::error) == static_cast<underlying>(Arch::xe64) + 1,
                  "Enum Arch is not sequential");
    return arch = static_cast<Arch>(static_cast<underlying>(arch) + 1);
}

ISPCTarget operator++(ISPCTarget &target, int dummy) {
    using underlying = std::underlying_type_t<ISPCTarget>;
    static_assert(static_cast<underlying>(ISPCTarget::host) == static_cast<underlying>(ISPCTarget::none) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::sse2_i32x4) == static_cast<underlying>(ISPCTarget::host) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::sse2_i32x8) ==
                      static_cast<underlying>(ISPCTarget::sse2_i32x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::sse41_i8x16) ==
                      static_cast<underlying>(ISPCTarget::sse2_i32x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::sse41_i16x8) ==
                      static_cast<underlying>(ISPCTarget::sse41_i8x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::sse41_i32x4) ==
                      static_cast<underlying>(ISPCTarget::sse41_i16x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::sse41_i32x8) ==
                      static_cast<underlying>(ISPCTarget::sse41_i32x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::sse4_i8x16) ==
                      static_cast<underlying>(ISPCTarget::sse41_i32x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::sse4_i16x8) ==
                      static_cast<underlying>(ISPCTarget::sse4_i8x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::sse4_i32x4) ==
                      static_cast<underlying>(ISPCTarget::sse4_i16x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::sse4_i32x8) ==
                      static_cast<underlying>(ISPCTarget::sse4_i32x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx1_i32x4) ==
                      static_cast<underlying>(ISPCTarget::sse4_i32x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx1_i32x8) ==
                      static_cast<underlying>(ISPCTarget::avx1_i32x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx1_i32x16) ==
                      static_cast<underlying>(ISPCTarget::avx1_i32x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx1_i64x4) ==
                      static_cast<underlying>(ISPCTarget::avx1_i32x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx2_i8x32) ==
                      static_cast<underlying>(ISPCTarget::avx1_i64x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx2_i16x16) ==
                      static_cast<underlying>(ISPCTarget::avx2_i8x32) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx2_i32x4) ==
                      static_cast<underlying>(ISPCTarget::avx2_i16x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx2_i32x8) ==
                      static_cast<underlying>(ISPCTarget::avx2_i32x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx2_i32x16) ==
                      static_cast<underlying>(ISPCTarget::avx2_i32x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx2_i64x4) ==
                      static_cast<underlying>(ISPCTarget::avx2_i32x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx2vnni_i32x4) ==
                      static_cast<underlying>(ISPCTarget::avx2_i64x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx2vnni_i32x8) ==
                      static_cast<underlying>(ISPCTarget::avx2vnni_i32x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx2vnni_i32x16) ==
                      static_cast<underlying>(ISPCTarget::avx2vnni_i32x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512skx_x4) ==
                      static_cast<underlying>(ISPCTarget::avx2vnni_i32x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512skx_x8) ==
                      static_cast<underlying>(ISPCTarget::avx512skx_x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512skx_x16) ==
                      static_cast<underlying>(ISPCTarget::avx512skx_x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512skx_x16_nozmm) ==
                      static_cast<underlying>(ISPCTarget::avx512skx_x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512skx_x32) ==
                      static_cast<underlying>(ISPCTarget::avx512skx_x16_nozmm) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512skx_x64) ==
                      static_cast<underlying>(ISPCTarget::avx512skx_x32) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512icl_x4) ==
                      static_cast<underlying>(ISPCTarget::avx512skx_x64) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512icl_x8) ==
                      static_cast<underlying>(ISPCTarget::avx512icl_x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512icl_x16) ==
                      static_cast<underlying>(ISPCTarget::avx512icl_x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512icl_x16_nozmm) ==
                      static_cast<underlying>(ISPCTarget::avx512icl_x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512icl_x32) ==
                      static_cast<underlying>(ISPCTarget::avx512icl_x16_nozmm) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512icl_x64) ==
                      static_cast<underlying>(ISPCTarget::avx512icl_x32) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512spr_x4) ==
                      static_cast<underlying>(ISPCTarget::avx512icl_x64) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512spr_x8) ==
                      static_cast<underlying>(ISPCTarget::avx512spr_x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512spr_x16) ==
                      static_cast<underlying>(ISPCTarget::avx512spr_x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512spr_x32) ==
                      static_cast<underlying>(ISPCTarget::avx512spr_x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512spr_x64) ==
                      static_cast<underlying>(ISPCTarget::avx512spr_x32) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512gnr_x4) ==
                      static_cast<underlying>(ISPCTarget::avx512spr_x64) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512gnr_x8) ==
                      static_cast<underlying>(ISPCTarget::avx512gnr_x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512gnr_x16) ==
                      static_cast<underlying>(ISPCTarget::avx512gnr_x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512gnr_x32) ==
                      static_cast<underlying>(ISPCTarget::avx512gnr_x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx512gnr_x64) ==
                      static_cast<underlying>(ISPCTarget::avx512gnr_x32) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx10_2dmr_x4) ==
                      static_cast<underlying>(ISPCTarget::avx512gnr_x64) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx10_2dmr_x8) ==
                      static_cast<underlying>(ISPCTarget::avx10_2dmr_x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx10_2dmr_x16) ==
                      static_cast<underlying>(ISPCTarget::avx10_2dmr_x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx10_2dmr_x32) ==
                      static_cast<underlying>(ISPCTarget::avx10_2dmr_x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx10_2dmr_x64) ==
                      static_cast<underlying>(ISPCTarget::avx10_2dmr_x32) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx10_2nvl_x4) ==
                      static_cast<underlying>(ISPCTarget::avx10_2dmr_x64) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx10_2nvl_x8) ==
                      static_cast<underlying>(ISPCTarget::avx10_2nvl_x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx10_2nvl_x16) ==
                      static_cast<underlying>(ISPCTarget::avx10_2nvl_x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx10_2nvl_x32) ==
                      static_cast<underlying>(ISPCTarget::avx10_2nvl_x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::avx10_2nvl_x64) ==
                      static_cast<underlying>(ISPCTarget::avx10_2nvl_x32) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::neon_i8x16) ==
                      static_cast<underlying>(ISPCTarget::avx10_2nvl_x64) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::neon_i8x32) ==
                      static_cast<underlying>(ISPCTarget::neon_i8x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::neon_i16x8) ==
                      static_cast<underlying>(ISPCTarget::neon_i8x32) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::neon_i16x16) ==
                      static_cast<underlying>(ISPCTarget::neon_i16x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::neon_i32x4) ==
                      static_cast<underlying>(ISPCTarget::neon_i16x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::neon_i32x8) ==
                      static_cast<underlying>(ISPCTarget::neon_i32x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::rvv_x4) == static_cast<underlying>(ISPCTarget::neon_i32x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::wasm_i32x4) == static_cast<underlying>(ISPCTarget::rvv_x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::xelp_x8) == static_cast<underlying>(ISPCTarget::wasm_i32x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::xelp_x16) == static_cast<underlying>(ISPCTarget::xelp_x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::xehpg_x8) == static_cast<underlying>(ISPCTarget::xelp_x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::xehpg_x16) == static_cast<underlying>(ISPCTarget::xehpg_x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::xehpc_x16) == static_cast<underlying>(ISPCTarget::xehpg_x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::xehpc_x32) == static_cast<underlying>(ISPCTarget::xehpc_x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::xelpg_x8) == static_cast<underlying>(ISPCTarget::xehpc_x32) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::xelpg_x16) == static_cast<underlying>(ISPCTarget::xelpg_x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::xe2hpg_x16) == static_cast<underlying>(ISPCTarget::xelpg_x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::xe2hpg_x32) ==
                      static_cast<underlying>(ISPCTarget::xe2hpg_x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::xe2lpg_x16) ==
                      static_cast<underlying>(ISPCTarget::xe2hpg_x32) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::xe2lpg_x32) ==
                      static_cast<underlying>(ISPCTarget::xe2lpg_x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::generic_i1x4) ==
                      static_cast<underlying>(ISPCTarget::xe2lpg_x32) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::generic_i1x8) ==
                      static_cast<underlying>(ISPCTarget::generic_i1x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::generic_i1x16) ==
                      static_cast<underlying>(ISPCTarget::generic_i1x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::generic_i1x32) ==
                      static_cast<underlying>(ISPCTarget::generic_i1x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::generic_i1x64) ==
                      static_cast<underlying>(ISPCTarget::generic_i1x32) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::generic_i8x16) ==
                      static_cast<underlying>(ISPCTarget::generic_i1x64) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::generic_i8x32) ==
                      static_cast<underlying>(ISPCTarget::generic_i8x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::generic_i16x8) ==
                      static_cast<underlying>(ISPCTarget::generic_i8x32) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::generic_i16x16) ==
                      static_cast<underlying>(ISPCTarget::generic_i16x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::generic_i32x4) ==
                      static_cast<underlying>(ISPCTarget::generic_i16x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::generic_i32x8) ==
                      static_cast<underlying>(ISPCTarget::generic_i32x4) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::generic_i32x16) ==
                      static_cast<underlying>(ISPCTarget::generic_i32x8) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::generic_i64x4) ==
                      static_cast<underlying>(ISPCTarget::generic_i32x16) + 1,
                  "Enum ISPCTarget is not sequential");
    static_assert(static_cast<underlying>(ISPCTarget::error) == static_cast<underlying>(ISPCTarget::generic_i64x4) + 1,
                  "Enum ISPCTarget is not sequential");
    return target = static_cast<ISPCTarget>(static_cast<underlying>(target) + 1);
}

Arch ParseArch(std::string arch) {
    if (arch == "x86") {
        return Arch::x86;
    } else if (arch == "x86_64" || arch == "x86-64") {
        return Arch::x86_64;
    } else if (arch == "arm") {
        return Arch::arm;
    } else if (arch == "aarch64") {
        return Arch::aarch64;
    } else if (arch == "riscv64") {
        return Arch::riscv64;
    } else if (arch == "wasm32") {
        return Arch::wasm32;
    } else if (arch == "wasm64") {
        return Arch::wasm64;
    } else if (arch == "xe64") {
        return Arch::xe64;
    }
    return Arch::error;
}

std::string ArchToString(Arch arch) {
    switch (arch) {
    case Arch::none:
        return "none";
    case Arch::x86:
        return "x86";
    case Arch::x86_64:
        return "x86-64";
    case Arch::arm:
        return "arm";
    case Arch::aarch64:
        return "aarch64";
    case Arch::riscv64:
        return "riscv64";
    case Arch::wasm32:
        return "wasm32";
    case Arch::wasm64:
        return "wasm64";
    case Arch::xe64:
        return "xe64";
    case Arch::error:
        return "error";
    default:
        // none and error are not supposed to be printed.
        Error(SourcePos(), "Invalid arch is processed");
        exit(1);
    }
    return "error";
}

ISPCTarget ParseISPCTarget(std::string target) {
    // TODO: ensure skx-i32x8 is not enabled and linked for earli LLVM version.

    // The first matching string for each target is the canonical way to name the target,
    // all other strings are aliases.
    if (target == "host") {
        return ISPCTarget::host;
    } else if (target == "sse2-i32x4" || target == "sse2") {
        return ISPCTarget::sse2_i32x4;
    } else if (target == "sse2-i32x8" || target == "sse2-x2") {
        return ISPCTarget::sse2_i32x8;
    } else if (target == "sse4-i8x16" || target == "sse4.2-i8x16") {
        return ISPCTarget::sse4_i8x16;
    } else if (target == "sse4-i16x8" || target == "sse4.2-i16x8") {
        return ISPCTarget::sse4_i16x8;
    } else if (target == "sse4-i32x4" || target == "sse4" || target == "sse4.2-i32x4") {
        return ISPCTarget::sse4_i32x4;
    } else if (target == "sse4-i32x8" || target == "sse4-x2" || target == "sse4x2" || target == "sse4.2-i32x8") {
        return ISPCTarget::sse4_i32x8;
    } else if (target == "sse4.1-i8x16") {
        return ISPCTarget::sse41_i8x16;
    } else if (target == "sse4.1-i16x8") {
        return ISPCTarget::sse41_i16x8;
    } else if (target == "sse4.1-i32x4") {
        return ISPCTarget::sse41_i32x4;
    } else if (target == "sse4.1-i32x8") {
        return ISPCTarget::sse41_i32x8;
    } else if (target == "avx1-i32x4") {
        return ISPCTarget::avx1_i32x4;
    } else if (target == "avx1-i32x8" || target == "avx" || target == "avx1") {
        return ISPCTarget::avx1_i32x8;
    } else if (target == "avx1-i64x4" || target == "avx-i64x4") {
        return ISPCTarget::avx1_i64x4;
    } else if (target == "avx1-i32x16" || target == "avx-x2" || target == "avx1-x2") {
        return ISPCTarget::avx1_i32x16;
    } else if (target == "avx2-i8x32") {
        return ISPCTarget::avx2_i8x32;
    } else if (target == "avx2-i16x16") {
        return ISPCTarget::avx2_i16x16;
    } else if (target == "avx2-i32x4") {
        return ISPCTarget::avx2_i32x4;
    } else if (target == "avx2-i32x8" || target == "avx2") {
        return ISPCTarget::avx2_i32x8;
    } else if (target == "avx2-i64x4") {
        return ISPCTarget::avx2_i64x4;
    } else if (target == "avx2-i32x16" || target == "avx2-x2") {
        return ISPCTarget::avx2_i32x16;
    } else if (target == "avx2vnni-i32x4") {
        return ISPCTarget::avx2vnni_i32x4;
    } else if (target == "avx2vnni-i32x8") {
        return ISPCTarget::avx2vnni_i32x8;
    } else if (target == "avx2vnni-i32x16") {
        return ISPCTarget::avx2vnni_i32x16;
    } else if (target == "avx512skx-x4" || target == "avx512skx-i32x4") {
        return ISPCTarget::avx512skx_x4;
    } else if (target == "avx512skx-x8" || target == "avx512skx-i32x8") {
        return ISPCTarget::avx512skx_x8;
    } else if (target == "avx512skx-x16" || target == "avx512skx-i32x16") {
        return ISPCTarget::avx512skx_x16;
    } else if (target == "avx512skx-x16-nozmm") {
        return ISPCTarget::avx512skx_x16_nozmm;
    } else if (target == "avx512skx-x32" || target == "avx512skx-i16x32") {
        return ISPCTarget::avx512skx_x32;
    } else if (target == "avx512skx-x64" || target == "avx512skx-i8x64") {
        return ISPCTarget::avx512skx_x64;
    } else if (target == "avx512icl-x4") {
        return ISPCTarget::avx512icl_x4;
    } else if (target == "avx512icl-x8") {
        return ISPCTarget::avx512icl_x8;
    } else if (target == "avx512icl-x16") {
        return ISPCTarget::avx512icl_x16;
    } else if (target == "avx512icl-x16-nozmm") {
        return ISPCTarget::avx512icl_x16_nozmm;
    } else if (target == "avx512icl-x32") {
        return ISPCTarget::avx512icl_x32;
    } else if (target == "avx512icl-x64") {
        return ISPCTarget::avx512icl_x64;
    } else if (target == "avx512spr-x4") {
        return ISPCTarget::avx512spr_x4;
    } else if (target == "avx512spr-x8") {
        return ISPCTarget::avx512spr_x8;
    } else if (target == "avx512spr-x16") {
        return ISPCTarget::avx512spr_x16;
    } else if (target == "avx512spr-x32") {
        return ISPCTarget::avx512spr_x32;
    } else if (target == "avx512spr-x64") {
        return ISPCTarget::avx512spr_x64;
    } else if (target == "avx512gnr-x4") {
        return ISPCTarget::avx512gnr_x4;
    } else if (target == "avx512gnr-x8") {
        return ISPCTarget::avx512gnr_x8;
    } else if (target == "avx512gnr-x16") {
        return ISPCTarget::avx512gnr_x16;
    } else if (target == "avx512gnr-x32") {
        return ISPCTarget::avx512gnr_x32;
    } else if (target == "avx512gnr-x64") {
        return ISPCTarget::avx512gnr_x64;
#if ISPC_LLVM_VERSION >= ISPC_LLVM_20_0
    } else if (target == "avx10.2dmr-x4") {
        return ISPCTarget::avx10_2dmr_x4;
    } else if (target == "avx10.2dmr-x8") {
        return ISPCTarget::avx10_2dmr_x8;
    } else if (target == "avx10.2dmr-x16") {
        return ISPCTarget::avx10_2dmr_x16;
    } else if (target == "avx10.2dmr-x32") {
        return ISPCTarget::avx10_2dmr_x32;
    } else if (target == "avx10.2dmr-x64") {
        return ISPCTarget::avx10_2dmr_x64;
#endif
#if ISPC_LLVM_VERSION >= ISPC_LLVM_22_0
    } else if (target == "avx10.2nvl-x4") {
        return ISPCTarget::avx10_2nvl_x4;
    } else if (target == "avx10.2nvl-x8") {
        return ISPCTarget::avx10_2nvl_x8;
    } else if (target == "avx10.2nvl-x16") {
        return ISPCTarget::avx10_2nvl_x16;
    } else if (target == "avx10.2nvl-x32") {
        return ISPCTarget::avx10_2nvl_x32;
    } else if (target == "avx10.2nvl-x64") {
        return ISPCTarget::avx10_2nvl_x64;
#endif
    } else if (target == "neon-i8x16") {
        return ISPCTarget::neon_i8x16;
    } else if (target == "neon-i8x32") {
        return ISPCTarget::neon_i8x32;
    } else if (target == "neon-i16x8") {
        return ISPCTarget::neon_i16x8;
    } else if (target == "neon-i16x16") {
        return ISPCTarget::neon_i16x16;
    } else if (target == "neon-i32x4" || target == "neon") {
        return ISPCTarget::neon_i32x4;
    } else if (target == "neon-i32x8") {
        return ISPCTarget::neon_i32x8;
    } else if (target == "rvv-x4") {
        return ISPCTarget::rvv_x4;
    } else if (target == "wasm-i32x4") {
        return ISPCTarget::wasm_i32x4;
    } else if (target == "xelp-x8") {
        return ISPCTarget::xelp_x8;
    } else if (target == "xelp-x16" || target == "xelp") {
        return ISPCTarget::xelp_x16;
    } else if (target == "xehpg-x8") {
        return ISPCTarget::xehpg_x8;
    } else if (target == "xehpg-x16") {
        return ISPCTarget::xehpg_x16;
    } else if (target == "xehpc-x16") {
        return ISPCTarget::xehpc_x16;
    } else if (target == "xehpc-x32") {
        return ISPCTarget::xehpc_x32;
    } else if (target == "xelpg-x8") {
        return ISPCTarget::xelpg_x8;
    } else if (target == "xelpg-x16") {
        return ISPCTarget::xelpg_x16;
    } else if (target == "xe2hpg-x16") {
        return ISPCTarget::xe2hpg_x16;
    } else if (target == "xe2hpg-x32") {
        return ISPCTarget::xe2hpg_x32;
    } else if (target == "xe2lpg-x16") {
        return ISPCTarget::xe2lpg_x16;
    } else if (target == "xe2lpg-x32") {
        return ISPCTarget::xe2lpg_x32;
    } else if (target == "generic-i1x4") {
        return ISPCTarget::generic_i1x4;
    } else if (target == "generic-i1x8") {
        return ISPCTarget::generic_i1x8;
    } else if (target == "generic-i1x16") {
        return ISPCTarget::generic_i1x16;
    } else if (target == "generic-i1x32") {
        return ISPCTarget::generic_i1x32;
    } else if (target == "generic-i1x64") {
        return ISPCTarget::generic_i1x64;
    } else if (target == "generic-i8x16") {
        return ISPCTarget::generic_i8x16;
    } else if (target == "generic-i8x32") {
        return ISPCTarget::generic_i8x32;
    } else if (target == "generic-i16x8") {
        return ISPCTarget::generic_i16x8;
    } else if (target == "generic-i16x16") {
        return ISPCTarget::generic_i16x16;
    } else if (target == "generic-i32x4") {
        return ISPCTarget::generic_i32x4;
    } else if (target == "generic-i32x8") {
        return ISPCTarget::generic_i32x8;
    } else if (target == "generic-i32x16") {
        return ISPCTarget::generic_i32x16;
    } else if (target == "generic-i64x4") {
        return ISPCTarget::generic_i64x4;
    }

    return ISPCTarget::error;
}

// Given a comma-delimited string with one or more compilation targets of
// the form "sse4-i32x4,avx2-i32x8", return a pair. First element of the pair is a vector
// of correctly parsed targets, second element of the pair is a strings with targets, which
// were not recognized.
std::pair<std::vector<ISPCTarget>, std::string> ParseISPCTargets(const char *target) {
    std::vector<ISPCTarget> targets;
    std::string error_target;
    const char *tstart = target;
    bool done = false;
    while (!done) {
        const char *tend = strchr(tstart, ',');
        if (tend == nullptr) {
            done = true;
            tend = strchr(tstart, '\0');
        }
        std::string target_string = std::string(tstart, tend);
        ISPCTarget target_parsed = ParseISPCTarget(target_string);
        if (target_parsed == ISPCTarget::error) {
            if (!error_target.empty()) {
                error_target += ",";
            }
            error_target += target_string;
        } else {
            targets.push_back(target_parsed);
        }
        tstart = tend + 1;
    }
    return std::make_pair(targets, error_target);
}

std::string ISPCTargetToString(ISPCTarget target) {
    switch (target) {
    case ISPCTarget::host:
        return "host";
    case ISPCTarget::sse2_i32x4:
        return "sse2-i32x4";
    case ISPCTarget::sse2_i32x8:
        return "sse2-i32x8";
    case ISPCTarget::sse41_i8x16:
        return "sse4.1-i8x16";
    case ISPCTarget::sse41_i16x8:
        return "sse4.1-i16x8";
    case ISPCTarget::sse41_i32x4:
        return "sse4.1-i32x4";
    case ISPCTarget::sse41_i32x8:
        return "sse4.1-i32x8";
    case ISPCTarget::sse4_i8x16:
        return "sse4.2-i8x16";
    case ISPCTarget::sse4_i16x8:
        return "sse4.2-i16x8";
    case ISPCTarget::sse4_i32x4:
        return "sse4.2-i32x4";
    case ISPCTarget::sse4_i32x8:
        return "sse4.2-i32x8";
    case ISPCTarget::avx1_i32x4:
        return "avx1-i32x4";
    case ISPCTarget::avx1_i32x8:
        return "avx1-i32x8";
    case ISPCTarget::avx1_i32x16:
        return "avx1-i32x16";
    case ISPCTarget::avx1_i64x4:
        return "avx1-i64x4";
    case ISPCTarget::avx2_i8x32:
        return "avx2-i8x32";
    case ISPCTarget::avx2_i16x16:
        return "avx2-i16x16";
    case ISPCTarget::avx2_i32x4:
        return "avx2-i32x4";
    case ISPCTarget::avx2_i32x8:
        return "avx2-i32x8";
    case ISPCTarget::avx2_i32x16:
        return "avx2-i32x16";
    case ISPCTarget::avx2_i64x4:
        return "avx2-i64x4";
    case ISPCTarget::avx2vnni_i32x4:
        return "avx2vnni-i32x4";
    case ISPCTarget::avx2vnni_i32x8:
        return "avx2vnni-i32x8";
    case ISPCTarget::avx2vnni_i32x16:
        return "avx2vnni-i32x16";
    case ISPCTarget::avx512skx_x4:
        return "avx512skx-x4";
    case ISPCTarget::avx512skx_x8:
        return "avx512skx-x8";
    case ISPCTarget::avx512skx_x16:
    case ISPCTarget::avx512skx_x16_nozmm:
        return "avx512skx-x16";
    case ISPCTarget::avx512skx_x32:
        return "avx512skx-x32";
    case ISPCTarget::avx512skx_x64:
        return "avx512skx-x64";
    case ISPCTarget::avx512icl_x4:
        return "avx512icl-x4";
    case ISPCTarget::avx512icl_x8:
        return "avx512icl-x8";
    case ISPCTarget::avx512icl_x16:
    case ISPCTarget::avx512icl_x16_nozmm:
        return "avx512icl-x16";
    case ISPCTarget::avx512icl_x32:
        return "avx512icl-x32";
    case ISPCTarget::avx512icl_x64:
        return "avx512icl-x64";
    case ISPCTarget::avx512spr_x4:
        return "avx512spr-x4";
    case ISPCTarget::avx512spr_x8:
        return "avx512spr-x8";
    case ISPCTarget::avx512spr_x16:
        return "avx512spr-x16";
    case ISPCTarget::avx512spr_x32:
        return "avx512spr-x32";
    case ISPCTarget::avx512spr_x64:
        return "avx512spr-x64";
    case ISPCTarget::avx512gnr_x4:
        return "avx512gnr-x4";
    case ISPCTarget::avx512gnr_x8:
        return "avx512gnr-x8";
    case ISPCTarget::avx512gnr_x16:
        return "avx512gnr-x16";
    case ISPCTarget::avx512gnr_x32:
        return "avx512gnr-x32";
    case ISPCTarget::avx512gnr_x64:
        return "avx512gnr-x64";
    case ISPCTarget::avx10_2dmr_x4:
        return "avx10.2dmr-x4";
    case ISPCTarget::avx10_2dmr_x8:
        return "avx10.2dmr-x8";
    case ISPCTarget::avx10_2dmr_x16:
        return "avx10.2dmr-x16";
    case ISPCTarget::avx10_2dmr_x32:
        return "avx10.2dmr-x32";
    case ISPCTarget::avx10_2dmr_x64:
        return "avx10.2dmr-x64";
    case ISPCTarget::avx10_2nvl_x4:
        return "avx10.2nvl-x4";
    case ISPCTarget::avx10_2nvl_x8:
        return "avx10.2nvl-x8";
    case ISPCTarget::avx10_2nvl_x16:
        return "avx10.2nvl-x16";
    case ISPCTarget::avx10_2nvl_x32:
        return "avx10.2nvl-x32";
    case ISPCTarget::avx10_2nvl_x64:
        return "avx10.2nvl-x64";
    case ISPCTarget::neon_i8x16:
        return "neon-i8x16";
    case ISPCTarget::neon_i8x32:
        return "neon-i8x32";
    case ISPCTarget::neon_i16x8:
        return "neon-i16x8";
    case ISPCTarget::neon_i16x16:
        return "neon-i16x16";
    case ISPCTarget::neon_i32x4:
        return "neon-i32x4";
    case ISPCTarget::neon_i32x8:
        return "neon-i32x8";
    case ISPCTarget::rvv_x4:
        return "rvv-x4";
    case ISPCTarget::wasm_i32x4:
        return "wasm-i32x4";
    case ISPCTarget::xelp_x8:
        return "xelp-x8";
    case ISPCTarget::xelp_x16:
        return "xelp-x16";
    case ISPCTarget::xehpg_x8:
        return "xehpg-x8";
    case ISPCTarget::xehpg_x16:
        return "xehpg-x16";
    case ISPCTarget::xehpc_x16:
        return "xehpc-x16";
    case ISPCTarget::xehpc_x32:
        return "xehpc-x32";
    case ISPCTarget::xelpg_x8:
        return "xelpg-x8";
    case ISPCTarget::xelpg_x16:
        return "xelpg-x16";
    case ISPCTarget::xe2hpg_x16:
        return "xe2hpg-x16";
    case ISPCTarget::xe2hpg_x32:
        return "xe2hpg-x32";
    case ISPCTarget::xe2lpg_x16:
        return "xe2lpg-x16";
    case ISPCTarget::xe2lpg_x32:
        return "xe2lpg-x32";
    case ISPCTarget::generic_i1x4:
        return "generic-i1x4";
    case ISPCTarget::generic_i1x8:
        return "generic-i1x8";
    case ISPCTarget::generic_i1x16:
        return "generic-i1x16";
    case ISPCTarget::generic_i1x32:
        return "generic-i1x32";
    case ISPCTarget::generic_i1x64:
        return "generic-i1x64";
    case ISPCTarget::generic_i8x16:
        return "generic-i8x16";
    case ISPCTarget::generic_i8x32:
        return "generic-i8x32";
    case ISPCTarget::generic_i16x8:
        return "generic-i16x8";
    case ISPCTarget::generic_i16x16:
        return "generic-i16x16";
    case ISPCTarget::generic_i32x4:
        return "generic-i32x4";
    case ISPCTarget::generic_i32x8:
        return "generic-i32x8";
    case ISPCTarget::generic_i32x16:
        return "generic-i32x16";
    case ISPCTarget::generic_i64x4:
        return "generic-i64x4";
    case ISPCTarget::none:
        return "none";
    case ISPCTarget::error:
        // Fall through
        ;
    }
    Error(SourcePos(), "Invalid ISPCTarget is processed");
    exit(1);
}

bool ISPCTargetIsX86(ISPCTarget target) {
    switch (target) {
    case ISPCTarget::sse2_i32x4:
    case ISPCTarget::sse2_i32x8:
    case ISPCTarget::sse41_i8x16:
    case ISPCTarget::sse41_i16x8:
    case ISPCTarget::sse41_i32x4:
    case ISPCTarget::sse41_i32x8:
    case ISPCTarget::sse4_i8x16:
    case ISPCTarget::sse4_i16x8:
    case ISPCTarget::sse4_i32x4:
    case ISPCTarget::sse4_i32x8:
    case ISPCTarget::avx1_i32x4:
    case ISPCTarget::avx1_i32x8:
    case ISPCTarget::avx1_i32x16:
    case ISPCTarget::avx1_i64x4:
    case ISPCTarget::avx2_i8x32:
    case ISPCTarget::avx2_i16x16:
    case ISPCTarget::avx2_i32x4:
    case ISPCTarget::avx2_i32x8:
    case ISPCTarget::avx2_i32x16:
    case ISPCTarget::avx2_i64x4:
    case ISPCTarget::avx2vnni_i32x4:
    case ISPCTarget::avx2vnni_i32x8:
    case ISPCTarget::avx2vnni_i32x16:
    case ISPCTarget::avx512skx_x4:
    case ISPCTarget::avx512skx_x8:
    case ISPCTarget::avx512skx_x16:
    case ISPCTarget::avx512skx_x16_nozmm:
    case ISPCTarget::avx512skx_x32:
    case ISPCTarget::avx512skx_x64:
    case ISPCTarget::avx512icl_x4:
    case ISPCTarget::avx512icl_x8:
    case ISPCTarget::avx512icl_x16:
    case ISPCTarget::avx512icl_x16_nozmm:
    case ISPCTarget::avx512icl_x32:
    case ISPCTarget::avx512icl_x64:
    case ISPCTarget::avx512spr_x4:
    case ISPCTarget::avx512spr_x8:
    case ISPCTarget::avx512spr_x16:
    case ISPCTarget::avx512spr_x32:
    case ISPCTarget::avx512spr_x64:
    case ISPCTarget::avx512gnr_x4:
    case ISPCTarget::avx512gnr_x8:
    case ISPCTarget::avx512gnr_x16:
    case ISPCTarget::avx512gnr_x32:
    case ISPCTarget::avx512gnr_x64:
    case ISPCTarget::avx10_2dmr_x4:
    case ISPCTarget::avx10_2dmr_x8:
    case ISPCTarget::avx10_2dmr_x16:
    case ISPCTarget::avx10_2dmr_x32:
    case ISPCTarget::avx10_2dmr_x64:
    case ISPCTarget::avx10_2nvl_x4:
    case ISPCTarget::avx10_2nvl_x8:
    case ISPCTarget::avx10_2nvl_x16:
    case ISPCTarget::avx10_2nvl_x32:
    case ISPCTarget::avx10_2nvl_x64:
        return true;
    default:
        return false;
    }
}

bool ISPCTargetIsNeon(ISPCTarget target) {
    switch (target) {
    case ISPCTarget::neon_i8x16:
    case ISPCTarget::neon_i8x32:
    case ISPCTarget::neon_i16x8:
    case ISPCTarget::neon_i16x16:
    case ISPCTarget::neon_i32x4:
    case ISPCTarget::neon_i32x8:
        return true;
    default:
        return false;
    }
}

bool ISPCTargetIsRiscV(ISPCTarget target) {
    switch (target) {
    case ISPCTarget::rvv_x4:
        return true;
    default:
        return false;
    }
}

bool ISPCTargetIsWasm(ISPCTarget target) {
    switch (target) {
    case ISPCTarget::wasm_i32x4:
        return true;
    default:
        return false;
    }
}

bool ISPCTargetIsGen(ISPCTarget target) {
    switch (target) {
    case ISPCTarget::xelp_x8:
    case ISPCTarget::xelp_x16:
    case ISPCTarget::xehpg_x8:
    case ISPCTarget::xehpg_x16:
    case ISPCTarget::xehpc_x16:
    case ISPCTarget::xehpc_x32:
    case ISPCTarget::xelpg_x8:
    case ISPCTarget::xelpg_x16:
    case ISPCTarget::xe2hpg_x16:
    case ISPCTarget::xe2hpg_x32:
    case ISPCTarget::xe2lpg_x16:
    case ISPCTarget::xe2lpg_x32:
        return true;
    default:
        return false;
    }
}

bool ISPCTargetIsGeneric(ISPCTarget target) {
    switch (target) {
    case ISPCTarget::generic_i1x4:
    case ISPCTarget::generic_i1x8:
    case ISPCTarget::generic_i1x16:
    case ISPCTarget::generic_i1x32:
    case ISPCTarget::generic_i1x64:
    case ISPCTarget::generic_i8x16:
    case ISPCTarget::generic_i8x32:
    case ISPCTarget::generic_i16x8:
    case ISPCTarget::generic_i16x16:
    case ISPCTarget::generic_i32x4:
    case ISPCTarget::generic_i32x8:
    case ISPCTarget::generic_i32x16:
    case ISPCTarget::generic_i64x4:
        return true;
    default:
        return false;
    }
}

TargetOS ParseOS(std::string os) {
    std::string supportedOses = g->target_registry->getSupportedOSes().c_str();
    if (supportedOses.find(os) == std::string::npos) {
        return TargetOS::error;
    }
    if (os == "windows") {
        return TargetOS::windows;
    } else if (os == "linux") {
        return TargetOS::linux;
    } else if (os == "custom_linux") {
        return TargetOS::custom_linux;
    } else if (os == "freebsd") {
        return TargetOS::freebsd;
    } else if (os == "macos") {
        return TargetOS::macos;
    } else if (os == "android") {
        return TargetOS::android;
    } else if (os == "ios") {
        return TargetOS::ios;
    } else if (os == "ps4") {
        return TargetOS::ps4;
    } else if (os == "ps5") {
        return TargetOS::ps5;
    } else if (os == "web") {
        return TargetOS::web;
    }
    return TargetOS::error;
}

std::string OSToString(TargetOS os) {
    switch (os) {
    case TargetOS::windows:
        return "Windows";
    case TargetOS::linux:
        return "Linux";
    case TargetOS::custom_linux:
        return "Linux (custom)";
    case TargetOS::freebsd:
        return "FreeBSD";
    case TargetOS::macos:
        return "macOS";
    case TargetOS::android:
        return "Android";
    case TargetOS::ios:
        return "iOS";
    case TargetOS::ps4:
        return "PS4";
    case TargetOS::ps5:
        return "PS5";
    case TargetOS::web:
        return "web";
    case TargetOS::error:
        return "error";
    }
    UNREACHABLE();
}

std::string OSToLowerString(TargetOS os) {
    switch (os) {
    case TargetOS::windows:
        return "windows";
    case TargetOS::linux:
        return "linux";
    case TargetOS::custom_linux:
        return "custom_linux";
    case TargetOS::freebsd:
        return "freebsd";
    case TargetOS::macos:
        return "macos";
    case TargetOS::android:
        return "android";
    case TargetOS::ios:
        return "ios";
    case TargetOS::ps4:
        return "ps4";
    case TargetOS::ps5:
        return "ps5";
    case TargetOS::web:
        return "web";
    case TargetOS::error:
        return "error";
    }
    UNREACHABLE();
}

TargetOS GetHostOS() {
#if defined(ISPC_HOST_IS_WINDOWS) && !defined(ISPC_WINDOWS_TARGET_OFF)
    return TargetOS::windows;
#elif defined(ISPC_HOST_IS_LINUX) && !defined(ISPC_LINUX_TARGET_OFF)
    return TargetOS::linux;
#elif defined(ISPC_HOST_IS_FREEBSD) && !defined(ISPC_FREEBSD_TARGET_OFF)
    return TargetOS::freebsd;
#elif defined(ISPC_HOST_IS_APPLE) && !defined(ISPC_MACOS_TARGET_OFF)
    return TargetOS::macos;
#else
    return TargetOS::error;
#endif
}
} // namespace ispc
