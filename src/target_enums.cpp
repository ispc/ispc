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
    } else if (arch == "genx32") {
        return Arch::genx32;
    } else if (arch == "genx64") {
        return Arch::genx64;
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
    case Arch::genx32:
        return "genx32";
    case Arch::genx64:
        return "genx64";
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
    } else if (target == "sse4-i8x16") {
        return ISPCTarget::sse4_i8x16;
    } else if (target == "sse4-i16x8") {
        return ISPCTarget::sse4_i16x8;
    } else if (target == "sse4-i32x4" || target == "sse4") {
        return ISPCTarget::sse4_i32x4;
    } else if (target == "sse4-i32x8" || target == "sse4-x2" || target == "sse4x2") {
        return ISPCTarget::sse4_i32x8;
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
    } else if (target == "avx512knl-i32x16") {
        return ISPCTarget::avx512knl_i32x16;
    } else if (target == "avx512skx-i32x16") {
        return ISPCTarget::avx512skx_i32x16;
    } else if (target == "avx512skx-i32x8") {
        return ISPCTarget::avx512skx_i32x8;
    } else if (target == "avx512skx-i8x64") {
        return ISPCTarget::avx512skx_i8x64;
    } else if (target == "avx512skx-i16x32") {
        return ISPCTarget::avx512skx_i16x32;
    } else if (target == "neon-i8x16") {
        return ISPCTarget::neon_i8x16;
    } else if (target == "neon-i16x8") {
        return ISPCTarget::neon_i8x16;
    } else if (target == "neon-i32x4" || target == "neon") {
        return ISPCTarget::neon_i32x4;
    } else if (target == "neon-i32x8") {
        return ISPCTarget::neon_i32x8;
    } else if (target == "wasm-i32x4") {
        return ISPCTarget::wasm_i32x4;
    } else if (target == "genx-x8") {
        return ISPCTarget::genx_x8;
    } else if (target == "genx-x16" || target == "genx") {
        return ISPCTarget::genx_x16;
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
        if (tend == NULL) {
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
    case ISPCTarget::sse4_i8x16:
        return "sse4-i8x16";
    case ISPCTarget::sse4_i16x8:
        return "sse4-i16x8";
    case ISPCTarget::sse4_i32x4:
        return "sse4-i32x4";
    case ISPCTarget::sse4_i32x8:
        return "sse4-i32x8";
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
    case ISPCTarget::avx512knl_i32x16:
        return "avx512knl-i32x16";
    case ISPCTarget::avx512skx_i32x8:
        return "avx512skx-i32x8";
    case ISPCTarget::avx512skx_i32x16:
        return "avx512skx-i32x16";
    case ISPCTarget::avx512skx_i8x64:
        return "avx512skx-i8x64";
    case ISPCTarget::avx512skx_i16x32:
        return "avx512skx-i16x32";
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
    case ISPCTarget::genx_x8:
        return "genx-x8";
    case ISPCTarget::genx_x16:
        return "genx-x16";
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
    case ISPCTarget::avx512knl_i32x16:
    case ISPCTarget::avx512skx_i32x8:
    case ISPCTarget::avx512skx_i32x16:
    case ISPCTarget::avx512skx_i8x64:
    case ISPCTarget::avx512skx_i16x32:
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
    case ISPCTarget::genx_x8:
    case ISPCTarget::genx_x16:
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
