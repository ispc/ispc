/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include <stdio.h>

#include "bitcode_lib.h"
#include "ispc.h"

#include "llvm/ADT/SmallString.h"
#include <llvm/Support/Path.h>

#define SSE2_TARGETS(M)                                                                                                \
    M(sse2_i32x4);                                                                                                     \
    M(sse2_i32x8);

#define SSE4_TARGETS(M)                                                                                                \
    M(sse4_i8x16)                                                                                                      \
    M(sse4_i16x8)                                                                                                      \
    M(sse4_i32x4)                                                                                                      \
    M(sse4_i32x8)

#define AVX1_TARGETS(M)                                                                                                \
    M(avx1_i32x8)                                                                                                      \
    M(avx1_i32x16)                                                                                                     \
    M(avx1_i64x4)

#define AVX2_TARGETS(M)                                                                                                \
    M(avx2_i8x32)                                                                                                      \
    M(avx2_i16x16)                                                                                                     \
    M(avx2_i32x4)                                                                                                      \
    M(avx2_i32x8)                                                                                                      \
    M(avx2_i32x16)                                                                                                     \
    M(avx2_i64x4)

#define AVX2VNNI_TARGETS(M)                                                                                            \
    M(avx2vnni_i32x4)                                                                                                  \
    M(avx2vnni_i32x8)                                                                                                  \
    M(avx2vnni_i32x16)

#define AVX512KNL_TARGETS(M) M(avx512knl_x16)

#define AVX512ICL_TARGETS(M)                                                                                           \
    M(avx512icl_x4)                                                                                                    \
    M(avx512icl_x8)                                                                                                    \
    M(avx512icl_x16)                                                                                                   \
    M(avx512icl_x32)                                                                                                   \
    M(avx512icl_x64)

#define AVX512SKX_TARGETS(M)                                                                                           \
    M(avx512skx_x4)                                                                                                    \
    M(avx512skx_x8)                                                                                                    \
    M(avx512skx_x16)                                                                                                   \
    M(avx512skx_x32)                                                                                                   \
    M(avx512skx_x64)

#define AVX512SPR_TARGETS(M)                                                                                           \
    M(avx512spr_x4)                                                                                                    \
    M(avx512spr_x8)                                                                                                    \
    M(avx512spr_x16)                                                                                                   \
    M(avx512spr_x32)                                                                                                   \
    M(avx512spr_x64)

#define X86_TARGETS(M)                                                                                                 \
    SSE2_TARGETS(M)                                                                                                    \
    SSE4_TARGETS(M)                                                                                                    \
    AVX1_TARGETS(M)                                                                                                    \
    AVX2_TARGETS(M)                                                                                                    \
    AVX2VNNI_TARGETS(M)                                                                                                \
    AVX512KNL_TARGETS(M)                                                                                               \
    AVX512SKX_TARGETS(M)                                                                                               \
    AVX512ICL_TARGETS(M)                                                                                               \
    AVX512SPR_TARGETS(M)

#define ARM_TARGETS(M)                                                                                                 \
    M(neon_i8x16)                                                                                                      \
    M(neon_i16x8)                                                                                                      \
    M(neon_i32x4)                                                                                                      \
    M(neon_i32x8)

#define AARCH64_TARGETS(M)                                                                                             \
    M(neon_i32x4)                                                                                                      \
    M(neon_i32x8)

#define XE_TARGETS(M)                                                                                                  \
    M(gen9_x8)                                                                                                         \
    M(gen9_x16)                                                                                                        \
    M(xelpg_x8)                                                                                                        \
    M(xelpg_x16)                                                                                                       \
    M(xelp_x8)                                                                                                         \
    M(xelp_x16)                                                                                                        \
    M(xehpg_x8)                                                                                                        \
    M(xehpg_x16)                                                                                                       \
    M(xehpc_x16)                                                                                                       \
    M(xehpc_x32)

#define WASM_TARGETS(M) M(wasm_i32x4)

#define TARGET_BITCODE_LIB(TARGET, OS_NAME, OS_TARGET, BIT, ARCH)                                                      \
    static BitcodeLib target_##TARGET##_##BIT##bit_##OS_NAME("builtins_target_" #TARGET "_" #BIT "bit_" #OS_NAME       \
                                                             ".bc",                                                    \
                                                             ISPCTarget::TARGET, TargetOS::OS_TARGET, Arch::ARCH)

#define TARGET_BITCODE_LIB_UNIX_X86_64(TARGET) TARGET_BITCODE_LIB(TARGET, unix, linux, 64, x86_64);
#define TARGET_BITCODE_LIB_UNIX_X86(TARGET) TARGET_BITCODE_LIB(TARGET, unix, linux, 32, x86);

#define TARGET_BITCODE_LIB_WINDOWS_X86_64(TARGET) TARGET_BITCODE_LIB(TARGET, windows, windows, 64, x86_64);
#define TARGET_BITCODE_LIB_WINDOWS_X86(TARGET) TARGET_BITCODE_LIB(TARGET, windows, windows, 32, x86);

#define TARGET_BITCODE_LIB_UNIX_ARM(TARGET) TARGET_BITCODE_LIB(TARGET, unix, linux, 32, arm);
#define TARGET_BITCODE_LIB_UNIX_AARCH64(TARGET) TARGET_BITCODE_LIB(TARGET, unix, linux, 64, aarch64);

#define TARGET_BITCODE_LIB_WINDOWS_ARM(TARGET) TARGET_BITCODE_LIB(TARGET, windows, windows, 32, arm);
#define TARGET_BITCODE_LIB_WINDOWS_AARCH64(TARGET) TARGET_BITCODE_LIB(TARGET, windows, windows, 64, aarch64);

#define TARGET_BITCODE_LIB_UNIX_XE(TARGET) TARGET_BITCODE_LIB(TARGET, unix, linux, 64, xe64);
#define TARGET_BITCODE_LIB_WINDOWS_XE(TARGET) TARGET_BITCODE_LIB(TARGET, windows, windows, 64, xe64);

#define TARGET_BITCODE_LIB_WEB_WASM32(TARGET) TARGET_BITCODE_LIB(TARGET, web, web, 32, wasm32);
#define TARGET_BITCODE_LIB_WEB_WASM64(TARGET) TARGET_BITCODE_LIB(TARGET, web, web, 64, wasm64);

using namespace ispc;

void printBinaryType() { printf("slim\n"); }

void initializeBinaryType(const char *ISPCExecutableAbsPath) {
    llvm::SmallString<128> includeDir(ISPCExecutableAbsPath);
    llvm::sys::path::remove_filename(includeDir);
    llvm::sys::path::remove_filename(includeDir);
    llvm::SmallString<128> shareDir(includeDir);
    // llvm::sys::path::append(includeDir, "include");
    // lParseInclude(includeDir.c_str());
    llvm::sys::path::append(shareDir, "share", "ispc");
    g->shareDirPath = std::string(shareDir.str());
}

// Dispatch bitcode_libs
// TODO! strangely enough this TargetOS has no sense and this dispatch lib is used on windows
static BitcodeLib dispatch("builtins_dispatch.bc", TargetOS::linux);
static BitcodeLib dispatch_macos("builtins_dispatch_macos.bc", TargetOS::macos);

// Common built_in bitcode_libs
#if defined(ISPC_LINUX_TARGET_ON) && defined(ISPC_ARM_ENABLED)
static BitcodeLib cpp_32_linux_armv7("builtins_cpp_32_linux_armv7.bc", TargetOS::linux, Arch::arm);
static BitcodeLib cpp_64_linux_aarch64("builtins_cpp_64_linux_aarch64.bc", TargetOS::linux, Arch::aarch64);
#endif // defined(ISPC_LINUX_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_LINUX_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib cpp_32_linux_i686("builtins_cpp_32_linux_i686.bc", TargetOS::linux, Arch::x86);
static BitcodeLib cpp_64_linux_x86_64("builtins_cpp_64_linux_x86_64.bc", TargetOS::linux, Arch::x86_64);
#endif // defined(ISPC_LINUX_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_ANDROID_TARGET_ON) && defined(ISPC_ARM_ENABLED)
static BitcodeLib cpp_32_android_armv7("builtins_cpp_32_android_armv7.bc", TargetOS::android, Arch::arm);
static BitcodeLib cpp_64_android_aarch64("builtins_cpp_64_android_aarch64.bc", TargetOS::android, Arch::aarch64);
#endif // defined(ISPC_ANDROID_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_ANDROID_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib cpp_32_android_i686("builtins_cpp_32_android_i686.bc", TargetOS::android, Arch::x86);
static BitcodeLib cpp_64_android_x86_64("builtins_cpp_64_android_x86_64.bc", TargetOS::android, Arch::x86_64);
#endif // defined(ISPC_ANDROID_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_FREEBSD_TARGET_ON) && defined(ISPC_ARM_ENABLED)
static BitcodeLib cpp_32_freebsd_armv7("builtins_cpp_32_freebsd_armv7.bc", TargetOS::freebsd, Arch::arm);
static BitcodeLib cpp_64_freebsd_aarch64("builtins_cpp_64_freebsd_aarch64.bc", TargetOS::freebsd, Arch::aarch64);
#endif // defined(ISPC_FREEBSD_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_FREEBSD_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib cpp_32_freebsd_i686("builtins_cpp_32_freebsd_i686.bc", TargetOS::freebsd, Arch::x86);
static BitcodeLib cpp_64_freebsd_x86_64("builtins_cpp_64_freebsd_x86_64.bc", TargetOS::freebsd, Arch::x86_64);
#endif // defined(ISPC_FREEBSD_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_ARM_ENABLED)
static BitcodeLib cpp_64_windows_aarch64("builtins_cpp_64_windows_aarch64.bc", TargetOS::windows, Arch::aarch64);
#endif // defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib cpp_32_windows_i686("builtins_cpp_32_windows_i686.bc", TargetOS::windows, Arch::x86);
static BitcodeLib cpp_64_windows_x86_64("builtins_cpp_64_windows_x86_64.bc", TargetOS::windows, Arch::x86_64);
#endif // defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_MACOS_TARGET_ON) && defined(ISPC_ARM_ENABLED)
static BitcodeLib cpp_64_macos_aarch64("builtins_cpp_64_macos_aarch64.bc", TargetOS::macos, Arch::aarch64);
#endif // defined(ISPC_MACOS_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_MACOS_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib cpp_64_macos_x86_64("builtins_cpp_64_macos_x86_64.bc", TargetOS::macos, Arch::x86_64);
#endif // defined(ISPC_MACOS_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_IOS_TARGET_ON) && defined(ISPC_ARM_ENABLED)
// TODO! ??
static BitcodeLib cpp_64_ios_arm64("builtins_cpp_64_ios_arm64.bc", TargetOS::ios, Arch::aarch64);
#endif // defined(ISPC_IOS_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_PS_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib cpp_64_ps4_x86_64("builtins_cpp_64_ps4_x86_64.bc", TargetOS::ps4, Arch::x86_64);
#endif // defined(ISPC_PS_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_LINUX_TARGET_ON) && defined(ISPC_XE_ENABLED)
static BitcodeLib cm_64_linux("builtins_cm_64.cpp", TargetOS::linux, Arch::xe64);
#endif // defined(ISPC_LINUX_TARGET_ON) && defined(ISPC_XE_ENABLED)

#if defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_XE_ENABLED)
static BitcodeLib cm_64_windows("builtins_cm_64.cpp", TargetOS::windows, Arch::xe64);
#endif // defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_XE_ENABLED)

#ifdef ISPC_WASM_ENABLED
static BitcodeLib cpp_32_web_wasm32("builtins_cpp_32_web_wasm32.cpp", TargetOS::web, Arch::wasm32);
static BitcodeLib cpp_64_web_wasm64("builtins_cpp_64_web_wasm64.cpp", TargetOS::web, Arch::wasm64);
#endif // ISPC_WASM_ENABLED

// Target-specific built-in bitcode_libs
#if defined(ISPC_UNIX_TARGET_ON) && defined(ISPC_X86_ENABLED)
X86_TARGETS(TARGET_BITCODE_LIB_UNIX_X86_64)
X86_TARGETS(TARGET_BITCODE_LIB_UNIX_X86)
#endif // defined(ISPC_UNIX_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_X86_ENABLED)
X86_TARGETS(TARGET_BITCODE_LIB_WINDOWS_X86_64)
X86_TARGETS(TARGET_BITCODE_LIB_WINDOWS_X86)
#endif // defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_UNIX_TARGET_ON) && defined(ISPC_ARM_ENABLED)
ARM_TARGETS(TARGET_BITCODE_LIB_UNIX_ARM)
AARCH64_TARGETS(TARGET_BITCODE_LIB_UNIX_AARCH64)
#endif // defined(ISPC_UNIX_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_ARM_ENABLED)
// TODO! all targets are supported on windows?
ARM_TARGETS(TARGET_BITCODE_LIB_WINDOWS_ARM)
AARCH64_TARGETS(TARGET_BITCODE_LIB_WINDOWS_AARCH64)
#endif // defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_LINUX_TARGET_ON) & defined(ISPC_XE_ENABLED)
XE_TARGETS(TARGET_BITCODE_LIB_UNIX_XE)
#endif // defined(ISPC_LINUX_TARGET_ON) & defined(ISPC_XE_ENABLED)

#if defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_XE_ENABLED)
XE_TARGETS(TARGET_BITCODE_LIB_WINDOWS_XE)
#endif // defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_XE_ENABLED)

#ifdef ISPC_WASM_ENABLED
WASM_TARGETS(TARGET_BITCODE_LIB_WEB_WASM32)
WASM_TARGETS(TARGET_BITCODE_LIB_WEB_WASM64)
#endif // ISPC_WASM_ENABLED
