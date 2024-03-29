/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include <stdio.h>

#include "bitcode_lib.h"
#include "ispc.h"

#include "llvm/ADT/SmallString.h"
#include <llvm/Support/Path.h>

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
static BitcodeLib dispatch("builtins-dispatch.bc", TargetOS::linux);
static BitcodeLib dispatch_macos("builtins-dispatch-macos.bc", TargetOS::macos);

// Common built-in bitcode_libs
#if defined(ISPC_LINUX_TARGET_ON) && defined(ISPC_ARM_ENABLED)
static BitcodeLib cpp_32_linux_armv7("builtins-cpp-32-linux-armv7.bc", TargetOS::linux, Arch::arm);
static BitcodeLib cpp_64_linux_aarch64("builtins-cpp-64-linux-aarch64.bc", TargetOS::linux, Arch::aarch64);
#endif // defined(ISPC_LINUX_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_LINUX_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib cpp_32_linux_i686("builtins-cpp-32-linux-i686.bc", TargetOS::linux, Arch::x86);
static BitcodeLib cpp_64_linux_x86_64("builtins-cpp-64-linux-x86_64.bc", TargetOS::linux, Arch::x86_64);
#endif // defined(ISPC_LINUX_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_ANDROID_TARGET_ON) && defined(ISPC_ARM_ENABLED)
static BitcodeLib cpp_32_android_armv7("builtins-cpp-32-android-armv7.bc", TargetOS::android, Arch::arm);
static BitcodeLib cpp_64_android_aarch64("builtins-cpp-64-android-aarch64.bc", TargetOS::android, Arch::aarch64);
#endif // defined(ISPC_ANDROID_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_ANDROID_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib cpp_32_android_i686("builtins-cpp-32-android-i686.bc", TargetOS::android, Arch::x86);
static BitcodeLib cpp_64_android_x86_64("builtins-cpp-64-android-x86_64.bc", TargetOS::android, Arch::x86_64);
#endif // defined(ISPC_ANDROID_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_FREEBSD_TARGET_ON) && defined(ISPC_ARM_ENABLED)
static BitcodeLib cpp_32_freebsd_armv7("builtins-cpp-32-freebsd-armv7.bc", TargetOS::freebsd, Arch::arm);
static BitcodeLib cpp_64_freebsd_aarch64("builtins-cpp-64-freebsd-aarch64.bc", TargetOS::freebsd, Arch::aarch64);
#endif // defined(ISPC_FREEBSD_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_FREEBSD_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib cpp_32_freebsd_i686("builtins-cpp-32-freebsd-i686.bc", TargetOS::freebsd, Arch::x86);
static BitcodeLib cpp_64_freebsd_x86_64("builtins-cpp-64-freebsd-x86_64.bc", TargetOS::freebsd, Arch::x86_64);
#endif // defined(ISPC_FREEBSD_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_ARM_ENABLED)
static BitcodeLib cpp_64_windows_aarch64("builtins-cpp-64-windows-aarch64.bc", TargetOS::windows, Arch::aarch64);
#endif // defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib cpp_32_windows_i686("builtins-cpp-32-windows-i686.bc", TargetOS::windows, Arch::x86);
static BitcodeLib cpp_64_windows_x86_64("builtins-cpp-64-windows-x86_64.bc", TargetOS::windows, Arch::x86_64);
#endif // defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_MACOS_TARGET_ON) && defined(ISPC_ARM_ENABLED)
static BitcodeLib cpp_64_macos_arm64("builtins-cpp-64-macos-arm64.bc", TargetOS::macos, Arch::aarch64);
#endif // defined(ISPC_MACOS_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_MACOS_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib cpp_64_macos_x86_64("builtins-cpp-64-macos-x86_64.bc", TargetOS::macos, Arch::x86_64);
#endif // defined(ISPC_MACOS_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_IOS_TARGET_ON) && defined(ISPC_ARM_ENABLED)
// TODO! ??
static BitcodeLib cpp_64_ios_arm64("builtins-cpp-64-ios-arm64.bc", TargetOS::ios, Arch::aarch64);
#endif // defined(ISPC_IOS_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_PS_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib cpp_64_ps4_x86_64("builtins-cpp-64-ps4-x86_64.bc", TargetOS::ps4, Arch::x86_64);
#endif // defined(ISPC_PS_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_LINUX_TARGET_ON) && defined(ISPC_XE_ENABLED)
static BitcodeLib cm_64_linux("builtins-cm-64.cpp", TargetOS::linux, Arch::xe64);
#endif // defined(ISPC_LINUX_TARGET_ON) && defined(ISPC_XE_ENABLED)

#if defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_XE_ENABLED)
static BitcodeLib cm_64_windows("builtins-cm-64.cpp", TargetOS::windows, Arch::xe64);
#endif // defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_XE_ENABLED)

#ifdef ISPC_WASM_ENABLED
static BitcodeLib cpp_32_web_wasm32("builtins-cpp-32-web-wasm32.cpp", TargetOS::web, Arch::wasm32);
static BitcodeLib cpp_64_web_wasm64("builtins-cpp-64-web-wasm64.cpp", TargetOS::web, Arch::wasm64);
#endif // ISPC_WASM_ENABLED

// Target-specific built-in bitcode_libs
// clang-format off
#if defined(ISPC_UNIX_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib target_sse2_i32x4_32bit_unix("builtins-target-sse2-i32x4-32bit-unix.bc", ISPCTarget::sse2_i32x4, TargetOS::linux, Arch::x86);
static BitcodeLib target_sse2_i32x8_32bit_unix("builtins-target-sse2-i32x8-32bit-unix.bc", ISPCTarget::sse2_i32x8, TargetOS::linux, Arch::x86);
static BitcodeLib target_sse2_i32x4_64bit_unix("builtins-target-sse2-i32x4-64bit-unix.bc", ISPCTarget::sse2_i32x4, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_sse2_i32x8_64bit_unix("builtins-target-sse2-i32x8-64bit-unix.bc", ISPCTarget::sse2_i32x8, TargetOS::linux, Arch::x86_64);

static BitcodeLib target_sse4_i8x16_32bit_unix("builtins-target-sse4-i8x16-32bit-unix.bc", ISPCTarget::sse4_i8x16, TargetOS::linux, Arch::x86);
static BitcodeLib target_sse4_i16x8_32bit_unix("builtins-target-sse4-i16x8-32bit-unix.bc", ISPCTarget::sse4_i16x8, TargetOS::linux, Arch::x86);
static BitcodeLib target_sse4_i32x4_32bit_unix("builtins-target-sse4-i32x4-32bit-unix.bc", ISPCTarget::sse4_i32x4, TargetOS::linux, Arch::x86);
static BitcodeLib target_sse4_i32x8_32bit_unix("builtins-target-sse4-i32x8-32bit-unix.bc", ISPCTarget::sse4_i32x8, TargetOS::linux, Arch::x86);
static BitcodeLib target_sse4_i8x16_64bit_unix("builtins-target-sse4-i8x16-64bit-unix.bc", ISPCTarget::sse4_i8x16, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_sse4_i16x8_64bit_unix("builtins-target-sse4-i16x8-64bit-unix.bc", ISPCTarget::sse4_i16x8, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_sse4_i32x4_64bit_unix("builtins-target-sse4-i32x4-64bit-unix.bc", ISPCTarget::sse4_i32x4, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_sse4_i32x8_64bit_unix("builtins-target-sse4-i32x8-64bit-unix.bc", ISPCTarget::sse4_i32x8, TargetOS::linux, Arch::x86_64);

static BitcodeLib target_avx1_i32x8_32bit_unix("builtins-target-avx1-i32x8-32bit-unix.bc", ISPCTarget::avx1_i32x8, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx1_i32x16_32bit_unix("builtins-target-avx1-i32x16-32bit-unix.bc", ISPCTarget::avx1_i32x16, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx1_i64x4_32bit_unix("builtins-target-avx1-i64x4-32bit-unix.bc", ISPCTarget::avx1_i64x4, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx1_i32x8_64bit_unix("builtins-target-avx1-i32x8-64bit-unix.bc", ISPCTarget::avx1_i32x8, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx1_i32x16_64bit_unix("builtins-target-avx1-i32x16-64bit-unix.bc", ISPCTarget::avx1_i32x16, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx1_i64x4_64bit_unix("builtins-target-avx1-i64x4-64bit-unix.bc", ISPCTarget::avx1_i64x4, TargetOS::linux, Arch::x86_64);

static BitcodeLib target_avx2_i8x32_32bit_unix("builtins-target-avx2-i8x32-32bit-unix.bc", ISPCTarget::avx2_i8x32, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx2_i16x16_32bit_unix("builtins-target-avx2-i16x16-32bit-unix.bc", ISPCTarget::avx2_i16x16, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx2_i32x4_32bit_unix("builtins-target-avx2-i32x4-32bit-unix.bc", ISPCTarget::avx2_i32x4, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx2_i32x8_32bit_unix("builtins-target-avx2-i32x8-32bit-unix.bc", ISPCTarget::avx2_i32x8, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx2_i32x16_32bit_unix("builtins-target-avx2-i32x16-32bit-unix.bc", ISPCTarget::avx2_i32x16, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx2_i64x4_32bit_unix("builtins-target-avx2-i64x4-32bit-unix.bc", ISPCTarget::avx2_i64x4, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx2_i8x32_64bit_unix("builtins-target-avx2-i8x32-64bit-unix.bc", ISPCTarget::avx2_i8x32, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx2_i16x16_64bit_unix("builtins-target-avx2-i16x16-64bit-unix.bc", ISPCTarget::avx2_i16x16, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx2_i32x4_64bit_unix("builtins-target-avx2-i32x4-64bit-unix.bc", ISPCTarget::avx2_i32x4, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx2_i32x8_64bit_unix("builtins-target-avx2-i32x8-64bit-unix.bc", ISPCTarget::avx2_i32x8, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx2_i32x16_64bit_unix("builtins-target-avx2-i32x16-64bit-unix.bc", ISPCTarget::avx2_i32x16, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx2_i64x4_64bit_unix("builtins-target-avx2-i64x4-64bit-unix.bc", ISPCTarget::avx2_i64x4, TargetOS::linux, Arch::x86_64);

static BitcodeLib target_avx2vnni_i32x4_32bit_unix("builtins-target-avx2vnni-i32x4-32bit-unix.bc", ISPCTarget::avx2vnni_i32x4, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx2vnni_i32x8_32bit_unix("builtins-target-avx2vnni-i32x8-32bit-unix.bc", ISPCTarget::avx2vnni_i32x8, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx2vnni_i32x16_32bit_unix("builtins-target-avx2vnni-i32x16-32bit-unix.bc", ISPCTarget::avx2vnni_i32x16, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx2vnni_i32x4_64bit_unix("builtins-target-avx2vnni-i32x4-64bit-unix.bc", ISPCTarget::avx2vnni_i32x4, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx2vnni_i32x8_64bit_unix("builtins-target-avx2vnni-i32x8-64bit-unix.bc", ISPCTarget::avx2vnni_i32x8, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx2vnni_i32x16_64bit_unix("builtins-target-avx2vnni-i32x16-64bit-unix.bc", ISPCTarget::avx2vnni_i32x16, TargetOS::linux, Arch::x86_64);

static BitcodeLib target_avx512knl_x16_32bit_unix("builtins-target-avx512knl-x16-32bit-unix.bc", ISPCTarget::avx512knl_x16, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512knl_x16_64bit_unix("builtins-target-avx512knl-x16-64bit-unix.bc", ISPCTarget::avx512knl_x16, TargetOS::linux, Arch::x86_64);

static BitcodeLib target_avx512skx_x4_32bit_unix("builtins-target-avx512skx-x4-32bit-unix.bc", ISPCTarget::avx512skx_x4, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512skx_x8_32bit_unix("builtins-target-avx512skx-x8-32bit-unix.bc", ISPCTarget::avx512skx_x8, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512skx_x16_32bit_unix("builtins-target-avx512skx-x16-32bit-unix.bc", ISPCTarget::avx512skx_x16, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512skx_x32_32bit_unix("builtins-target-avx512skx-x32-32bit-unix.bc", ISPCTarget::avx512skx_x32, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512skx_x64_32bit_unix("builtins-target-avx512skx-x64-32bit-unix.bc", ISPCTarget::avx512skx_x64, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512skx_x4_64bit_unix("builtins-target-avx512skx-x4-64bit-unix.bc", ISPCTarget::avx512skx_x4, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx512skx_x8_64bit_unix("builtins-target-avx512skx-x8-64bit-unix.bc", ISPCTarget::avx512skx_x8, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx512skx_x16_64bit_unix("builtins-target-avx512skx-x16-64bit-unix.bc", ISPCTarget::avx512skx_x16, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx512skx_x32_64bit_unix("builtins-target-avx512skx-x32-64bit-unix.bc", ISPCTarget::avx512skx_x32, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx512skx_x64_64bit_unix("builtins-target-avx512skx-x64-64bit-unix.bc", ISPCTarget::avx512skx_x64, TargetOS::linux, Arch::x86_64);

static BitcodeLib target_avx512icl_x4_32bit_unix("builtins-target-avx512icl-x4-32bit-unix.bc", ISPCTarget::avx512icl_x4, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512icl_x8_32bit_unix("builtins-target-avx512icl-x8-32bit-unix.bc", ISPCTarget::avx512icl_x8, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512icl_x16_32bit_unix("builtins-target-avx512icl-x16-32bit-unix.bc", ISPCTarget::avx512icl_x16, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512icl_x32_32bit_unix("builtins-target-avx512icl-x32-32bit-unix.bc", ISPCTarget::avx512icl_x32, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512icl_x64_32bit_unix("builtins-target-avx512icl-x64-32bit-unix.bc", ISPCTarget::avx512icl_x64, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512icl_x4_64bit_unix("builtins-target-avx512icl-x4-64bit-unix.bc", ISPCTarget::avx512icl_x4, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx512icl_x8_64bit_unix("builtins-target-avx512icl-x8-64bit-unix.bc", ISPCTarget::avx512icl_x8, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx512icl_x16_64bit_unix("builtins-target-avx512icl-x16-64bit-unix.bc", ISPCTarget::avx512icl_x16, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx512icl_x32_64bit_unix("builtins-target-avx512icl-x32-64bit-unix.bc", ISPCTarget::avx512icl_x32, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx512icl_x64_64bit_unix("builtins-target-avx512icl-x64-64bit-unix.bc", ISPCTarget::avx512icl_x64, TargetOS::linux, Arch::x86_64);

static BitcodeLib target_avx512spr_x4_32bit_unix("builtins-target-avx512spr-x4-32bit-unix.bc", ISPCTarget::avx512spr_x4, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512spr_x8_32bit_unix("builtins-target-avx512spr-x8-32bit-unix.bc", ISPCTarget::avx512spr_x8, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512spr_x16_32bit_unix("builtins-target-avx512spr-x16-32bit-unix.bc", ISPCTarget::avx512spr_x16, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512spr_x32_32bit_unix("builtins-target-avx512spr-x32-32bit-unix.bc", ISPCTarget::avx512spr_x32, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512spr_x64_32bit_unix("builtins-target-avx512spr-x64-32bit-unix.bc", ISPCTarget::avx512spr_x64, TargetOS::linux, Arch::x86);
static BitcodeLib target_avx512spr_x4_64bit_unix("builtins-target-avx512spr-x4-64bit-unix.bc", ISPCTarget::avx512spr_x4, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx512spr_x8_64bit_unix("builtins-target-avx512spr-x8-64bit-unix.bc", ISPCTarget::avx512spr_x8, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx512spr_x16_64bit_unix("builtins-target-avx512spr-x16-64bit-unix.bc", ISPCTarget::avx512spr_x16, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx512spr_x32_64bit_unix("builtins-target-avx512spr-x32-64bit-unix.bc", ISPCTarget::avx512spr_x32, TargetOS::linux, Arch::x86_64);
static BitcodeLib target_avx512spr_x64_64bit_unix("builtins-target-avx512spr-x64-64bit-unix.bc", ISPCTarget::avx512spr_x64, TargetOS::linux, Arch::x86_64);
#endif // defined(ISPC_UNIX_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_X86_ENABLED)
static BitcodeLib target_sse2_i32x4_32bit_windows("builtins-target-sse2-i32x4-32bit-windows.bc", ISPCTarget::sse2_i32x4, TargetOS::windows, Arch::x86);
static BitcodeLib target_sse2_i32x8_32bit_windows("builtins-target-sse2-i32x8-32bit-windows.bc", ISPCTarget::sse2_i32x8, TargetOS::windows, Arch::x86);
static BitcodeLib target_sse2_i32x4_64bit_windows("builtins-target-sse2-i32x4-64bit-windows.bc", ISPCTarget::sse2_i32x4, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_sse2_i32x8_64bit_windows("builtins-target-sse2-i32x8-64bit-windows.bc", ISPCTarget::sse2_i32x8, TargetOS::windows, Arch::x86_64);

static BitcodeLib target_sse4_i8x16_32bit_windows("builtins-target-sse4-i8x16-32bit-windows.bc", ISPCTarget::sse4_i8x16, TargetOS::windows, Arch::x86);
static BitcodeLib target_sse4_i16x8_32bit_windows("builtins-target-sse4-i16x8-32bit-windows.bc", ISPCTarget::sse4_i16x8, TargetOS::windows, Arch::x86);
static BitcodeLib target_sse4_i32x4_32bit_windows("builtins-target-sse4-i32x4-32bit-windows.bc", ISPCTarget::sse4_i32x4, TargetOS::windows, Arch::x86);
static BitcodeLib target_sse4_i32x8_32bit_windows("builtins-target-sse4-i32x8-32bit-windows.bc", ISPCTarget::sse4_i32x8, TargetOS::windows, Arch::x86);
static BitcodeLib target_sse4_i8x16_64bit_windows("builtins-target-sse4-i8x16-64bit-windows.bc", ISPCTarget::sse4_i8x16, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_sse4_i16x8_64bit_windows("builtins-target-sse4-i16x8-64bit-windows.bc", ISPCTarget::sse4_i16x8, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_sse4_i32x4_64bit_windows("builtins-target-sse4-i32x4-64bit-windows.bc", ISPCTarget::sse4_i32x4, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_sse4_i32x8_64bit_windows("builtins-target-sse4-i32x8-64bit-windows.bc", ISPCTarget::sse4_i32x8, TargetOS::windows, Arch::x86_64);

static BitcodeLib target_avx1_i32x8_32bit_windows("builtins-target-avx1-i32x8-32bit-windows.bc", ISPCTarget::avx1_i32x8, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx1_i32x16_32bit_windows("builtins-target-avx1-i32x16-32bit-windows.bc", ISPCTarget::avx1_i32x16, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx1_i64x4_32bit_windows("builtins-target-avx1-i64x4-32bit-windows.bc", ISPCTarget::avx1_i64x4, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx1_i32x8_64bit_windows("builtins-target-avx1-i32x8-64bit-windows.bc", ISPCTarget::avx1_i32x8, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx1_i32x16_64bit_windows("builtins-target-avx1-i32x16-64bit-windows.bc", ISPCTarget::avx1_i32x16, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx1_i64x4_64bit_windows("builtins-target-avx1-i64x4-64bit-windows.bc", ISPCTarget::avx1_i64x4, TargetOS::windows, Arch::x86_64);

static BitcodeLib target_avx2_i8x32_32bit_windows("builtins-target-avx2-i8x32-32bit-windows.bc", ISPCTarget::avx2_i8x32, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx2_i16x16_32bit_windows("builtins-target-avx2-i16x16-32bit-windows.bc", ISPCTarget::avx2_i16x16, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx2_i32x4_32bit_windows("builtins-target-avx2-i32x4-32bit-windows.bc", ISPCTarget::avx2_i32x4, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx2_i32x8_32bit_windows("builtins-target-avx2-i32x8-32bit-windows.bc", ISPCTarget::avx2_i32x8, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx2_i32x16_32bit_windows("builtins-target-avx2-i32x16-32bit-windows.bc", ISPCTarget::avx2_i32x16, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx2_i64x4_32bit_windows("builtins-target-avx2-i64x4-32bit-windows.bc", ISPCTarget::avx2_i64x4, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx2_i8x32_64bit_windows("builtins-target-avx2-i8x32-64bit-windows.bc", ISPCTarget::avx2_i8x32, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx2_i16x16_64bit_windows("builtins-target-avx2-i16x16-64bit-windows.bc", ISPCTarget::avx2_i16x16, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx2_i32x4_64bit_windows("builtins-target-avx2-i32x4-64bit-windows.bc", ISPCTarget::avx2_i32x4, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx2_i32x8_64bit_windows("builtins-target-avx2-i32x8-64bit-windows.bc", ISPCTarget::avx2_i32x8, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx2_i32x16_64bit_windows("builtins-target-avx2-i32x16-64bit-windows.bc", ISPCTarget::avx2_i32x16, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx2_i64x4_64bit_windows("builtins-target-avx2-i64x4-64bit-windows.bc", ISPCTarget::avx2_i64x4, TargetOS::windows, Arch::x86_64);

static BitcodeLib target_avx512knl_x16_32bit_windows("builtins-target-avx512knl-x16-32bit-windows.bc", ISPCTarget::avx512knl_x16, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx512skx_x4_32bit_windows("builtins-target-avx512skx-x4-32bit-windows.bc", ISPCTarget::avx512skx_x4, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx512skx_x8_32bit_windows("builtins-target-avx512skx-x8-32bit-windows.bc", ISPCTarget::avx512skx_x8, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx512skx_x16_32bit_windows("builtins-target-avx512skx-x16-32bit-windows.bc", ISPCTarget::avx512skx_x16, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx512skx_x32_32bit_windows("builtins-target-avx512skx-x32-32bit-windows.bc", ISPCTarget::avx512skx_x32, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx512skx_x64_32bit_windows("builtins-target-avx512skx-x64-32bit-windows.bc", ISPCTarget::avx512skx_x64, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx512knl_x16_64bit_windows("builtins-target-avx512knl-x16-64bit-windows.bc", ISPCTarget::avx512knl_x16, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx512skx_x4_64bit_windows("builtins-target-avx512skx-x4-64bit-windows.bc", ISPCTarget::avx512skx_x4, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx512skx_x8_64bit_windows("builtins-target-avx512skx-x8-64bit-windows.bc", ISPCTarget::avx512skx_x8, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx512skx_x16_64bit_windows("builtins-target-avx512skx-x16-64bit-windows.bc", ISPCTarget::avx512skx_x16, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx512skx_x32_64bit_windows("builtins-target-avx512skx-x32-64bit-windows.bc", ISPCTarget::avx512skx_x32, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx512skx_x64_64bit_windows("builtins-target-avx512skx-x64-64bit-windows.bc", ISPCTarget::avx512skx_x64, TargetOS::windows, Arch::x86_64);

static BitcodeLib target_avx512spr_x4_32bit_windows("builtins-target-avx512spr-x4-32bit-windows.bc", ISPCTarget::avx512spr_x4, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx512spr_x8_32bit_windows("builtins-target-avx512spr-x8-32bit-windows.bc", ISPCTarget::avx512spr_x8, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx512spr_x16_32bit_windows("builtins-target-avx512spr-x16-32bit-windows.bc", ISPCTarget::avx512spr_x16, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx512spr_x32_32bit_windows("builtins-target-avx512spr-x32-32bit-windows.bc", ISPCTarget::avx512spr_x32, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx512spr_x64_32bit_windows("builtins-target-avx512spr-x64-32bit-windows.bc", ISPCTarget::avx512spr_x64, TargetOS::windows, Arch::x86);
static BitcodeLib target_avx512spr_x4_64bit_windows("builtins-target-avx512spr-x4-64bit-windows.bc", ISPCTarget::avx512spr_x4, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx512spr_x8_64bit_windows("builtins-target-avx512spr-x8-64bit-windows.bc", ISPCTarget::avx512spr_x8, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx512spr_x16_64bit_windows("builtins-target-avx512spr-x16-64bit-windows.bc", ISPCTarget::avx512spr_x16, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx512spr_x32_64bit_windows("builtins-target-avx512spr-x32-64bit-windows.bc", ISPCTarget::avx512spr_x32, TargetOS::windows, Arch::x86_64);
static BitcodeLib target_avx512spr_x64_64bit_windows("builtins-target-avx512spr-x64-64bit-windows.bc", ISPCTarget::avx512spr_x64, TargetOS::windows, Arch::x86_64);
#endif // defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_X86_ENABLED)

#if defined(ISPC_UNIX_TARGET_ON) && defined(ISPC_ARM_ENABLED)
static BitcodeLib target_neon_i8x16_32bit_unix("builtins-target-neon-i8x16-32bit-unix.bc", ISPCTarget::neon_i8x16, TargetOS::linux, Arch::arm);
static BitcodeLib target_neon_i16x8_32bit_unix("builtins-target-neon-i16x8-32bit-unix.bc", ISPCTarget::neon_i16x8, TargetOS::linux, Arch::arm);
static BitcodeLib target_neon_i32x4_32bit_unix("builtins-target-neon-i32x4-32bit-unix.bc", ISPCTarget::neon_i32x4, TargetOS::linux, Arch::arm);
static BitcodeLib target_neon_i32x8_32bit_unix("builtins-target-neon-i32x8-32bit-unix.bc", ISPCTarget::neon_i32x8, TargetOS::linux, Arch::arm);

static BitcodeLib target_neon_i32x4_64bit_unix("builtins-target-neon-i32x4-64bit-unix.bc", ISPCTarget::neon_i32x4, TargetOS::linux, Arch::aarch64);
static BitcodeLib target_neon_i32x8_64bit_unix("builtins-target-neon-i32x8-64bit-unix.bc", ISPCTarget::neon_i32x8, TargetOS::linux, Arch::aarch64);
#endif // defined(ISPC_UNIX_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_ARM_ENABLED)
// TODO! all targets are supported on windows?
static BitcodeLib target_neon_i8x16_32bit_windows("builtins-target-neon-i8x16-32bit-windows.bc", ISPCTarget::neon_i8x16, TargetOS::windows, Arch::arm);
static BitcodeLib target_neon_i16x8_32bit_windows("builtins-target-neon-i16x8-32bit-windows.bc", ISPCTarget::neon_i16x8, TargetOS::windows, Arch::arm);
static BitcodeLib target_neon_i32x4_32bit_windows("builtins-target-neon-i32x4-32bit-windows.bc", ISPCTarget::neon_i32x4, TargetOS::windows, Arch::arm);
static BitcodeLib target_neon_i32x8_32bit_windows("builtins-target-neon-i32x8-32bit-windows.bc", ISPCTarget::neon_i32x8, TargetOS::windows, Arch::arm);

static BitcodeLib target_neon_i32x4_64bit_windows("builtins-target-neon-i32x4-64bit-windows.bc", ISPCTarget::neon_i32x4, TargetOS::windows, Arch::aarch64);
static BitcodeLib target_neon_i32x8_64bit_windows("builtins-target-neon-i32x8-64bit-windows.bc", ISPCTarget::neon_i32x8, TargetOS::windows, Arch::aarch64);
#endif // defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_ARM_ENABLED)

#if defined(ISPC_LINUX_TARGET_ON) & defined(ISPC_XE_ENABLED)
static BitcodeLib target_gen9_x8_64bit_unix("builtins-target-gen9-x8-64bit-unix.bc", ISPCTarget::gen9_x8, TargetOS::linux, Arch::xe64);
static BitcodeLib target_gen9_x16_64bit_unix("builtins-target-gen9-x16-64bit-unix.bc", ISPCTarget::gen9_x16, TargetOS::linux, Arch::xe64);
static BitcodeLib target_xelpg_x8_64bit_unix("builtins-target-xelpg-x8-64bit-unix.bc", ISPCTarget::xelpg_x8, TargetOS::linux, Arch::xe64);
static BitcodeLib target_xelpg_x16_64bit_unix("builtins-target-xelpg-x16-64bit-unix.bc", ISPCTarget::xelpg_x16, TargetOS::linux, Arch::xe64);
static BitcodeLib target_xelp_x8_64bit_unix("builtins-target-xelp-x8-64bit-unix.bc", ISPCTarget::xelp_x8, TargetOS::linux, Arch::xe64);
static BitcodeLib target_xelp_x16_64bit_unix("builtins-target-xelp-x16-64bit-unix.bc", ISPCTarget::xelp_x16, TargetOS::linux, Arch::xe64);
static BitcodeLib target_xehpg_x8_64bit_unix("builtins-target-xehpg-x8-64bit-unix.bc", ISPCTarget::xehpg_x8, TargetOS::linux, Arch::xe64);
static BitcodeLib target_xehpg_x16_64bit_unix("builtins-target-xehpg-x16-64bit-unix.bc", ISPCTarget::xehpg_x16, TargetOS::linux, Arch::xe64);
static BitcodeLib target_xehpc_x16_64bit_unix("builtins-target-xehpc-x16-64bit-unix.bc", ISPCTarget::xehpc_x16, TargetOS::linux, Arch::xe64);
static BitcodeLib target_xehpc_x32_64bit_unix("builtins-target-xehpc-x32-64bit-unix.bc", ISPCTarget::xehpc_x32, TargetOS::linux, Arch::xe64);
#endif // defined(ISPC_LINUX_TARGET_ON) & defined(ISPC_XE_ENABLED)

#if defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_XE_ENABLED)
static BitcodeLib target_gen9_x8_64bit_windows("builtins-target-gen9-x8-64bit-windows.bc", ISPCTarget::gen9_x8, TargetOS::windows, Arch::xe64);
static BitcodeLib target_gen9_x16_64bit_windows("builtins-target-gen9-x16-64bit-windows.bc", ISPCTarget::gen9_x16, TargetOS::windows, Arch::xe64);
static BitcodeLib target_xelpg_x8_64bit_windows("builtins-target-xelpg-x8-64bit-windows.bc", ISPCTarget::xelpg_x8, TargetOS::windows, Arch::xe64);
static BitcodeLib target_xelpg_x16_64bit_windows("builtins-target-xelpg-x16-64bit-windows.bc", ISPCTarget::xelpg_x16, TargetOS::windows, Arch::xe64);
static BitcodeLib target_xelp_x8_64bit_windows("builtins-target-xelp-x8-64bit-windows.bc", ISPCTarget::xelp_x8, TargetOS::windows, Arch::xe64);
static BitcodeLib target_xelp_x16_64bit_windows("builtins-target-xelp-x16-64bit-windows.bc", ISPCTarget::xelp_x16, TargetOS::windows, Arch::xe64);
static BitcodeLib target_xehpg_x8_64bit_windows("builtins-target-xehpg-x8-64bit-windows.bc", ISPCTarget::xehpg_x8, TargetOS::windows, Arch::xe64);
static BitcodeLib target_xehpg_x16_64bit_windows("builtins-target-xehpg-x16-64bit-windows.bc", ISPCTarget::xehpg_x16, TargetOS::windows, Arch::xe64);
static BitcodeLib target_xehpc_x16_64bit_windows("builtins-target-xehpc-x16-64bit-windows.bc", ISPCTarget::xehpc_x16, TargetOS::windows, Arch::xe64);
static BitcodeLib target_xehpc_x32_64bit_windows("builtins-target-xehpc-x32-64bit-windows.bc", ISPCTarget::xehpc_x32, TargetOS::windows, Arch::xe64);
#endif // defined(ISPC_WINDOWS_TARGET_ON) && defined(ISPC_XE_ENABLED)

#ifdef ISPC_WASM_ENABLED
static BitcodeLib target_wasm_i32x4_32bit_web("builtins-target-wasm-i32x4-32bit-web.bc", ISPCTarget::wasm_i32x4, TargetOS::web, Arch::wasm32);
static BitcodeLib target_wasm_i32x4_64bit_web("builtins-target-wasm-i32x4-64bit-web.bc", ISPCTarget::wasm_i32x4, TargetOS::web, Arch::wasm64);
#endif // ISPC_WASM_ENABLED
// clang-format on
