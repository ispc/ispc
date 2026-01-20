// RUN: not %{ispc} %s --target=sse4.1-i32x4 --enable-llvm-intrinsics -o /dev/null 2>&1 | FileCheck %s
// REQUIRES: X86_ENABLED

uniform int gMode = 3;

float bad_round(float v) {
    return @llvm.x86.sse41.round.ps(v, gMode);
}

// CHECK: Error: Immediate argument 2 to "llvm.x86.sse41.round.ps" must be a
// CHECK: compile-time constant.
