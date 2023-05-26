// This test checks that ISPC_FP16_SUPPORTED macro is ON for XE and ARM platforms only.

// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK_FP_16,CHECK
// RUN: %{ispc} %s --target=neon-i8x16 --arch=arm --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK_FP_16,CHECK
// RUN: %{ispc} %s --target=neon-i32x4 --arch=aarch64 --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK_FP_16,CHECK
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86 --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK_FP_NO_16,CHECK
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK_FP_NO_16,CHECK

// REQUIRES: XE_ENABLED
// REQUIRES: X86_ENABLED
// We do not support ARM targets on Windows.
// Not every ISPC build on macOS supports ARM targets.
// They are supported only if macOS SDK supports them.
// REQUIRES: ARM_ENABLED && !WINDOWS_HOST && (!MACOS_HOST || MACOS_ARM_ENABLED)

// CHECK_FP_16: void @test_fp16_only(
// CHECK_FP_NO_16-NOT: void @test_fp16_only(
// CHECK: void @test(
#ifdef ISPC_FP16_SUPPORTED
export void test_fp16_only(uniform float16 out[], uniform float16 a) {
  out[programIndex] = a;
}
#endif

export void test(uniform float16 out[], uniform float16 a) {
  out[programIndex] = a;
}

