/*
  Copyright (c) 2019-2024, Intel Corporation

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

Arch ParseArch(std::string arch) {
    if (arch == "x86") {
        return Arch::x86;
    } else if (arch == "x86_64" || arch == "x86-64") {
        return Arch::x86_64;
    } else if (arch == "arm") {
        return Arch::arm;
    } else if (arch == "aarch64") {
        return Arch::aarch64;
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
    } else if (target == "avx512knl-x16" || target == "avx512knl-i32x16") {
        return ISPCTarget::avx512knl_x16;
    } else if (target == "avx512skx-x4" || target == "avx512skx-i32x4") {
        return ISPCTarget::avx512skx_x4;
    } else if (target == "avx512skx-x8" || target == "avx512skx-i32x8") {
        return ISPCTarget::avx512skx_x8;
    } else if (target == "avx512skx-x16" || target == "avx512skx-i32x16") {
        return ISPCTarget::avx512skx_x16;
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
    } else if (target == "neon-i8x16") {
        return ISPCTarget::neon_i8x16;
    } else if (target == "neon-i16x8") {
        return ISPCTarget::neon_i16x8;
    } else if (target == "neon-i32x4" || target == "neon") {
        return ISPCTarget::neon_i32x4;
    } else if (target == "neon-i32x8") {
        return ISPCTarget::neon_i32x8;
    } else if (target == "wasm-i32x4") {
        return ISPCTarget::wasm_i32x4;
    } else if (target == "gen9-x8") {
        return ISPCTarget::gen9_x8;
    } else if (target == "gen9-x16" || target == "gen9") {
        return ISPCTarget::gen9_x16;
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
    case ISPCTarget::avx512knl_x16:
        return "avx512knl-x16";
    case ISPCTarget::avx512skx_x4:
        return "avx512skx-x4";
    case ISPCTarget::avx512skx_x8:
        return "avx512skx-x8";
    case ISPCTarget::avx512skx_x16:
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
    case ISPCTarget::neon_i8x16:
        return "neon-i8x16";
    case ISPCTarget::neon_i16x8:
        return "neon-i16x8";
    case ISPCTarget::neon_i32x4:
        return "neon-i32x4";
    case ISPCTarget::neon_i32x8:
        return "neon-i32x8";
    case ISPCTarget::wasm_i32x4:
        return "wasm-i32x4";
    case ISPCTarget::gen9_x8:
        return "gen9-x8";
    case ISPCTarget::gen9_x16:
        return "gen9-x16";
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
    case ISPCTarget::none:
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
    case ISPCTarget::avx512knl_x16:
    case ISPCTarget::avx512skx_x4:
    case ISPCTarget::avx512skx_x8:
    case ISPCTarget::avx512skx_x16:
    case ISPCTarget::avx512skx_x32:
    case ISPCTarget::avx512skx_x64:
    case ISPCTarget::avx512icl_x4:
    case ISPCTarget::avx512icl_x8:
    case ISPCTarget::avx512icl_x16:
    case ISPCTarget::avx512icl_x32:
    case ISPCTarget::avx512icl_x64:
    case ISPCTarget::avx512spr_x4:
    case ISPCTarget::avx512spr_x8:
    case ISPCTarget::avx512spr_x16:
    case ISPCTarget::avx512spr_x32:
    case ISPCTarget::avx512spr_x64:
        return true;
    default:
        return false;
    }
}

bool ISPCTargetIsNeon(ISPCTarget target) {
    switch (target) {
    case ISPCTarget::neon_i8x16:
    case ISPCTarget::neon_i16x8:
    case ISPCTarget::neon_i32x4:
    case ISPCTarget::neon_i32x8:
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
    case ISPCTarget::gen9_x8:
    case ISPCTarget::gen9_x16:
    case ISPCTarget::xelp_x8:
    case ISPCTarget::xelp_x16:
    case ISPCTarget::xehpg_x8:
    case ISPCTarget::xehpg_x16:
    case ISPCTarget::xehpc_x16:
    case ISPCTarget::xehpc_x32:
    case ISPCTarget::xelpg_x8:
    case ISPCTarget::xelpg_x16:
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
