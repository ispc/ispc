// RUN: %{ispc} %s --target=xehpc-x16 --device=pvc -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefixes=CHECK_ALL,CHECK_FP64_SUPPORTED %s
// RUN: %{ispc} %s --target=xelp-x16 --device=tgllp -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefixes=CHECK_ALL,CHECK_FP64_NOT_SUPPORTED %s

// REQUIRES: XE_ENABLED

// CHECK_ALL-LABEL:       @test

#ifdef ISPC_TARGET_HAS_FP64_SUPPORT

// CHECK_FP64_SUPPORTED:  store <16 x double> %a_load_broadcast{{[0-9]*}}
export void test(uniform double out[], uniform double a) {

#else

// CHECK_FP64_NOT_SUPPORTED:  store <16 x float> %a_load_broadcast{{[0-9]*}}
export void test(uniform float out[], uniform float a) {

#endif

  out[programIndex] = a;
}
