// RUN: %{ispc}  %s --emit-asm --target=sse4-i32x4 -o - | FileCheck %s
// REQUIRES: X86_ENABLED
// The main loop should be something like:
// movups  (%r9,%rcx), %xmm0
// movups  (%r8,%rcx), %xmm1
// mulps   %xmm0, %xmm1
// movups  %xmm1, (%rdi,%rcx)

// CHECK: movups  ([[REG_1:%[a-z0-9]+]],[[REG_2:%[a-z0-9]+]]), [[REG_v1:%xmm[0-9]+]]
// CHECK: movups  ([[REG_3:%[a-z0-9]+]],[[REG_2]]), [[REG_v2:%xmm[0-9]+]]
// CHECK: mulps   [[REG_v1]], [[REG_v2]]
// CHECK: movups  [[REG:%xmm[0-9]+]], ([[REG_4:%[a-z0-9]+]],[[REG_2]])
export void bench_main(
  uniform float a[],
  uniform int N) {
  uniform float *b = a + N;
  uniform float *c = a + 2 * N;
  foreach(i = 0...N) {
    a[i] = b[i] * c[i];
  }
}
