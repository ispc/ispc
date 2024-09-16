// RUN: %{ispc} --no-discard-value-names --target=avx2-i32x8 --x86-asm-syntax=intel --emit-asm -o - %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: {{.*}}LBB0_3: {{.*}} %foreach_full_body
// CHECK-NEXT: {{.*}}  =>This Inner Loop Header: Depth=1
// CHECK-NEXT:         vaddps  [[NEW:ymm.*]], [[ONE:ymm.*]], ymmword ptr [[[BASE:r.*]] + 4*[[COUNTER:r.*]]]
// CHECK-NEXT:         vmovups ymmword ptr [[[BASE]] + 4*[[COUNTER]]], [[NEW]]
// CHECK-NEXT:         add     [[COUNTER]], 8
// CHECK-NEXT:         cmp     [[COUNTER]], [[N:r.*]]
// CHECK-NEXT:         jb      {{.*}}LBB0_3

unmasked void foo(uniform float Data[], const uniform int N) {
        foreach(i = 0 ... N) {
                Data[i] = Data[i] + 1;
        }
}
