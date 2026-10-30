// Check that GatherCoalescePass sign extends negative constant offsets with
// 32-bit addressing: the lanes read top[-1], top[0], top[-3], top[-2], ...,
// so the coalesced load must start 4 bytes past p, not 2^32 + 4.

// RUN: %{ispc} %s --target=sse2-i32x4 --addressing=32 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=sse4-i32x4 --addressing=32 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx1-i32x8 --addressing=32 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 --addressing=32 --opt=disable-gathers --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x16 --addressing=32 --opt=disable-gathers --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: @k___
// CHECK-NOT: getelementptr i8, ptr %p, i64 {{[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]}}
// CHECK: getelementptr i8, ptr %p, i64 4{{$}}
// CHECK-NOT: getelementptr i8, ptr %p, i64 {{[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]}}
// CHECK: ret void
void k(uniform float p[], uniform float out[]) {
    uniform float *uniform top = p + programCount;
    foreach (j = 0 ... programCount) {
        int t = -(j ^ 1);
        out[j] = top[t];
    }
}
