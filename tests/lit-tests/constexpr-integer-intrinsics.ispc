// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
//
// RUN: %{ispc} %s --target=host --nostdlib --nowrap --werror -O0 --emit-llvm-text -o %t.ll
// RUN: %{ispc} %s --target=host --nostdlib --nowrap --werror -O2 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=host --nostdlib --nowrap --werror -O2 -DALIAS_FIRST --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=host --nostdlib --nowrap --werror -O0 --off-phase=pre:last --emit-llvm-text -o %t.ll

// Native LLVM intrinsics retain their canonical ISPC signatures. The operation
// selects signed or unsigned saturation; an unsigned library wrapper converts
// the canonical signed result back to the appropriate unsigned type.
static_assert(@llvm.sadd.sat((uniform int8)127, (uniform int8)1) == (uniform int8)127);
static_assert(@llvm.ssub.sat((uniform int8)-128, (uniform int8)1) == (uniform int8)-128);
static_assert((uniform uint8)@llvm.uadd.sat((uniform uint8)250, (uniform uint8)10) == 255);
static_assert((uniform uint8)@llvm.usub.sat((uniform uint8)1, (uniform uint8)2) == 0);
static_assert(@llvm.sadd.sat((uniform int16)32767, (uniform int16)1) == (uniform int16)32767);
static_assert(@llvm.ssub.sat((uniform int16)-32768, (uniform int16)1) == (uniform int16)-32768);
static_assert((uniform uint16)@llvm.uadd.sat((uniform uint16)65535, (uniform uint16)1) == 65535);
static_assert((uniform uint16)@llvm.usub.sat((uniform uint16)1, (uniform uint16)65535) == 0);
static_assert(@llvm.sadd.sat(2147483647, 1) == 2147483647);
static_assert(@llvm.ssub.sat((uniform int32)0x80000000u, 1) == (uniform int32)0x80000000u);
static_assert((uniform uint32)@llvm.uadd.sat(0xffffffffu, 1u) == 0xffffffffu);
static_assert((uniform uint32)@llvm.usub.sat(1u, 0xffffffffu) == 0);
static_assert(@llvm.sadd.sat(9223372036854775807ll, 1ll) == 9223372036854775807ll);
static_assert(@llvm.ssub.sat((uniform int64)0x8000000000000000ull, 1ll) == (uniform int64)0x8000000000000000ull);
static_assert((uniform uint64)@llvm.uadd.sat(0xffffffffffffffffull, 1ull) == 0xffffffffffffffffull);
static_assert((uniform uint64)@llvm.usub.sat(1ull, 0xffffffffffffffffull) == 0);

// Typed ISPC aliases retain logical short-vector shape and unsignedness.
constexpr uniform int16<3> signed_a = {32767, -32767, 10};
constexpr uniform int16<3> signed_b = {1, -2, -20};
constexpr uniform int16<3> signed_sum = @llvm.ispc.sadd.sat(signed_a, signed_b);
constexpr uniform int16<3> signed_difference = @llvm.ispc.ssub.sat(signed_a, signed_b);
static_assert(signed_sum[0] == 32767 && signed_sum[1] == (uniform int16)-32768 && signed_sum[2] == -10);
static_assert(signed_difference[0] == 32766 && signed_difference[1] == -32765 && signed_difference[2] == 30);
constexpr uniform uint8<TARGET_WIDTH> bytes = (uniform uint8<TARGET_WIDTH>)250;
constexpr uniform uint8<TARGET_WIDTH> increment = (uniform uint8<TARGET_WIDTH>)10;
constexpr uniform uint8<TARGET_WIDTH> sum = @llvm.ispc.uadd.sat(bytes, increment);
constexpr uniform uint8<TARGET_WIDTH> difference = @llvm.ispc.usub.sat(increment, bytes);
static_assert(sum[0] == 255 && sum[TARGET_WIDTH - 1] == 255 && difference[TARGET_WIDTH - 1] == 0);

// Pack the low bit of each logical lane; storage padding contributes no bits.
static_assert(@llvm.ispc.packmask((uniform bool<1>)true) == 1);
static_assert((uniform uint64)@llvm.ispc.packmask((uniform bool<64>)true) == 0xffffffffffffffffull);
constexpr uniform uint64 alternating_mask(uniform int count, uniform int parity) {
    uniform uint64 result = 0;
    for (uniform int i = 0; i < count; ++i) {
        if ((i & 1) == parity) result |= 1ull << i;
    }
    return result;
}
constexpr varying bool even_lane = (programIndex & 1) == 0;
static_assert((uniform uint64)@llvm.ispc.packmask(even_lane) == alternating_mask(programCount, 0));
#if ISPC_MASK_BITS > 1
constexpr UIntMaskType integer_mask = (UIntMaskType)(programIndex * 3);
static_assert((uniform uint64)@llvm.ispc.packmask(integer_mask) == alternating_mask(programCount, 1));
#endif

static inline uniform int identify(uniform uint32<TARGET_WIDTH> x) { return 2; }
static inline uniform int identify(varying int32 x) { return 3; }

// Raw and typed calls share an LLVM declaration. Cache frontend signatures
// independently, regardless of which call creates the declaration first.
uniform int cache_types(uniform uint32<TARGET_WIDTH> packed, varying int32 lanes) {
#ifdef ALIAS_FIRST
    uniform int packed_result = identify(@llvm.ispc.uadd.sat(packed, packed));
    uniform int lane_result = identify(@llvm.uadd.sat(lanes, lanes));
#else
    uniform int lane_result = identify(@llvm.uadd.sat(lanes, lanes));
    uniform int packed_result = identify(@llvm.ispc.uadd.sat(packed, packed));
#endif
    return packed_result + 10 * lane_result;
}
// CHECK-LABEL: @cache_types___
// CHECK: ret i32 32
