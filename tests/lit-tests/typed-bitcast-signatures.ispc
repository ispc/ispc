// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
//
// Uniform short vectors and varying scalars can share LLVM types. Cache their
// ISPC signatures separately, independently of which call is seen first.
// RUN: %{ispc} %s --target=neon-i32x4 --arch=aarch64 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=neon-i32x4 --arch=aarch64 --nostdlib --nowrap -DSHORT_FIRST --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=neon-i32x8 --arch=aarch64 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=neon-i32x8 --arch=aarch64 --nostdlib --nowrap -DSHORT_FIRST --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=neon-i16x16 --arch=aarch64 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=neon-i16x16 --arch=aarch64 --nostdlib --nowrap -DSHORT_FIRST --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=neon-i32x4 --arch=aarch64 --nostdlib --nowrap -O0 --off-phase=all --emit-llvm-text -o - | FileCheck %s --check-prefix=LOGICAL
// REQUIRES: ARM_ENABLED

static inline uniform int identify(uniform int32<TARGET_WIDTH> x) { return 1; }
static inline uniform int identify(uniform uint32<TARGET_WIDTH> x) { return 2; }
static inline uniform int identify(varying int32 x) { return 3; }
static inline uniform int identify(varying uint32 x) { return 4; }

#ifdef SHORT_FIRST
#define FIRST_RESULT identify(@llvm.ispc.bitcast(packed, (uniform uint32<TARGET_WIDTH>)0))
#define SECOND_RESULT identify(@llvm.ispc.bitcast(lanes, (varying int32)0))
#else
#define FIRST_RESULT identify(@llvm.ispc.bitcast(lanes, (varying int32)0))
#define SECOND_RESULT identify(@llvm.ispc.bitcast(packed, (uniform uint32<TARGET_WIDTH>)0))
#endif

uniform int check_signatures(uniform float<TARGET_WIDTH> packed, varying float lanes) {
    uniform int first = FIRST_RESULT;
    uniform int second = SECOND_RESULT;
    uniform int signed_short = identify(@llvm.ispc.bitcast(packed, (uniform int32<TARGET_WIDTH>)0));
    uniform int unsigned_varying = identify(@llvm.ispc.bitcast(lanes, (varying uint32)0));
    return first + second + signed_short * 10 + unsigned_varying * 100;
}
// CHECK-LABEL: @check_signatures___
// CHECK: ret i32 415

uniform int32<TARGET_WIDTH> signed_source(uniform float<TARGET_WIDTH> packed) {
    return @llvm.ispc.bitcast(@llvm.ispc.bitcast(packed, (uniform uint32<TARGET_WIDTH>)0),
                              (uniform int32<TARGET_WIDTH>)0);
}

uniform uint32 merge_halves(uniform uint16<2> halves) {
    return @llvm.ispc.bitcast(halves, (uniform uint32)0);
}
// LOGICAL-LABEL: @merge_halves___
// LOGICAL: bitcast <2 x i16> {{.*}} to i32

uniform int16<3> merge_bytes(uniform uint8<6> bytes) {
    return @llvm.ispc.bitcast(bytes, (uniform int16<3>)0);
}
// LOGICAL-LABEL: @merge_bytes___
// LOGICAL: [[PACKED:%.*]] = bitcast <6 x i8> {{.*}} to <3 x i16>
// LOGICAL: shufflevector <3 x i16> [[PACKED]], <3 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
