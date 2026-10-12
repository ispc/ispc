// This test checks that ISPC_TARGET_HAS_FP16_SUPPORT macro is ON for 32-bit ARM.

// RUN: %{ispc} %s --target=neon-i8x16 --arch=arm --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK_FP_16,CHECK

// Windows and Mac do not support 32 bit ARM, so test on Linux only.
// REQUIRES: ARM32_ENABLED && LINUX_HOST

// CHECK_FP_16: void @test_fp16_only(
// CHECK: void @test(
#ifdef ISPC_TARGET_HAS_FP16_SUPPORT
export void test_fp16_only(uniform float16 out[], uniform float16 a) {
  out[programIndex] = a;
}
#endif

export void test(uniform float16 out[], uniform float16 a) {
  out[programIndex] = a;
}
