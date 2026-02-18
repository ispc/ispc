// RUN: %{ispc} %s --target=sse4.1-i32x4 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s
// REQUIRES: X86_ENABLED

constexpr uniform int kRoundMode = 3;

constexpr uniform int mode_func() { return 3; }

float round_constexpr(float v) {
    return @llvm.x86.sse41.round.ps(v, kRoundMode);
}

// CHECK-LABEL: @round_constexpr
// CHECK: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 3)

float round_constexpr_func(float v) {
    return @llvm.x86.sse41.round.ps(v, mode_func());
}

// CHECK-LABEL: @round_constexpr_func
// CHECK: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 3)
