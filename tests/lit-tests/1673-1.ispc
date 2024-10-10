// RUN: %{ispc}  %s --emit-asm --target=sse4-i32x4 -o - | FileCheck %s
// REQUIRES: X86_ENABLED && LLVM_16_0+ && !LLVM_17_0+
// The main loop should be something like:
// movups  (%r9,%rcx), %xmm0
// movups  (%r8,%rcx), %xmm1
// mulps   %xmm0, %xmm1
// movups  %xmm1, (%rdi,%rcx)

// CHECK-LABEL: {{.*}}LBB0_3: {{.*}}
// CHECK-NEXT: movups {{.*}}
// CHECK-NEXT: movups {{.*}}
// CHECK-NEXT: mulps {{.*}}
// CHECK-NEXT: movups {{.*}}
unmasked void bench_main(
  uniform float a[],
  uniform int N) {
  uniform float *b = a + N;
  uniform float *c = a + 2 * N;
  foreach(i = 0...N) {
    a[i] = b[i] * c[i];
  }
}
