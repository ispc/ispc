// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
//
// RUN: %{ispc} %s --target=host --nostdlib --nowrap --werror --emit-llvm-text -o - | FileCheck %s

// Preserve FP payloads when changing both lane count and element type.
constexpr uniform uint32<4> input = {0x7f800135u, 0x80000000u, 0x7fc00246u, 0x3ff00000u};
constexpr uniform float<4> float_view = @llvm.ispc.bitcast(input, (uniform float<4>)0);
constexpr uniform double<2> double_view = @llvm.ispc.bitcast(float_view, (uniform double<2>)0);
constexpr uniform uint64<2> wide_bits = @llvm.ispc.bitcast(double_view, (uniform uint64<2>)0);
static_assert(wide_bits[0] == 0x800000007f800135ull && wide_bits[1] == 0x3ff000007fc00246ull);
constexpr uniform uint32<4> output = @llvm.ispc.bitcast(double_view, (uniform uint32<4>)0);
static_assert(output[0] == input[0] && output[1] == input[1] && output[2] == input[2] && output[3] == input[3]);

// Only declared lanes contribute bits, even when both vectors are padded.
constexpr uniform uint8<6> bytes = {0x00, 0x80, 0x34, 0x12, 0xff, 0xff};
constexpr uniform int16<3> signed_words = @llvm.ispc.bitcast(bytes, (uniform int16<3>)0);
static_assert(signed_words[0] == -32768 && signed_words[1] == 0x1234 && signed_words[2] == -1);
constexpr uniform uint8<6> bytes_again = @llvm.ispc.bitcast(signed_words, (uniform uint8<6>)0);
static_assert(bytes_again[1] == 0x80 && bytes_again[5] == 0xff);

// Identical LLVM signatures retain the distinct signed and unsigned ISPC results.
static_assert((@llvm.ispc.bitcast(bytes, (uniform int16<3>)0))[0] < 0);
static_assert((@llvm.ispc.bitcast(bytes, (uniform uint16<3>)0))[0] > 0);
static_assert(@llvm.ispc.bitcast(0xffffffffu, (uniform int32)0) == -1);
static_assert(@llvm.ispc.bitcast(-1, (uniform uint32)0) > 0);

// A uniform scalar and a padded short vector can have the same logical width.
constexpr uniform uint16<2> halves = @llvm.ispc.bitcast(0xfffe0180u, (uniform uint16<2>)0);
static_assert(halves[0] == 0x0180 && halves[1] == 0xfffe);
static_assert(@llvm.ispc.bitcast(halves, (uniform uint32)0) == 0xfffe0180u);

// Varying atomics keep one independent payload per program instance.
varying uint32 varying_roundtrip() {
    constexpr varying uint32 bits = (programIndex & 1) != 0 ? 0x7f800135u : 0x80000000u;
    constexpr varying float view = @llvm.ispc.bitcast(bits, (varying float)0);
    constexpr varying uint32 result = @llvm.ispc.bitcast(view, (varying uint32)0);
    return result;
}
// CHECK-LABEL: @varying_roundtrip
// CHECK: ret <{{[0-9]+}} x i32> <i32 -2147483648, i32 2139095349{{.*}}>
