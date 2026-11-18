// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// SSE2 emits scalar min/max pairs for 64-bit clamp and retains vector min/max elsewhere.
// RUN: %{ispc} %s --target=sse2-i32x4 --nowrap -O2 --emit-llvm-text -o - | FileCheck %s --check-prefixes=SSE2,COMMON
// RUN: %{ispc} %s --target=avx2-i32x4 --nowrap -O2 --emit-llvm-text -o - | FileCheck %s --check-prefixes=AVX2,COMMON
// REQUIRES: X86_ENABLED

// SSE2-LABEL: @signed_clamp___
// SSE2: call i64 @llvm.smax.i64
// SSE2-COUNT-4: call i64 @llvm.smin.i64
varying int64 signed_clamp(varying int64 value, varying int64 low, varying int64 high) {
    return clamp(value, low, high);
}

// SSE2-LABEL: @unsigned_clamp___
// SSE2: call i64 @llvm.umax.i64
// SSE2-COUNT-4: call i64 @llvm.umin.i64
varying uint64 unsigned_clamp(varying uint64 value, varying uint64 low, varying uint64 high) {
    return clamp(value, low, high);
}

// SSE2-LABEL: @swapped_unsigned_clamp___
// SSE2: %[[LOWER:[^ ]+]] = call i64 @llvm.umax.i64
// SSE2-NEXT: {{.*}}call i64 @llvm.umin.i64(i64 %[[LOWER]],
// AVX2-LABEL: @swapped_unsigned_clamp___
// AVX2: call <4 x i64> @llvm.umax.v4i64
// AVX2: call <4 x i64> @llvm.umin.v4i64
varying uint64 swapped_unsigned_clamp(varying uint64 value, varying uint64 low, varying uint64 high) {
    return @llvm.umin(high, @llvm.umax(value, low));
}

// COMMON-LABEL: @standalone_min___
// COMMON: call <4 x i64> @llvm.smin.v4i64
varying int64 standalone_min(varying int64 a, varying int64 b) { return @llvm.smin(a, b); }

// COMMON-LABEL: @shared_max___
// COMMON: call <4 x i64> @llvm.smax.v4i64
// COMMON: call <4 x i64> @llvm.smin.v4i64
varying int64 shared_max(varying int64 value, varying int64 low, varying int64 high) {
    varying int64 lower = @llvm.smax(value, low);
    return @llvm.smin(lower, high) ^ lower;
}
