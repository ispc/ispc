// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
//
// RUN: %{ispc} %s --target=host --nostdlib --nowrap --werror -O0 --emit-llvm-text -o %t.ll
// RUN: %{ispc} %s --target=host --nostdlib --nowrap --werror -O2 --emit-llvm-text -o %t.ll
// RUN: %{ispc} %s --target=host --nostdlib --nowrap --werror --wrap-signed-int -DTEST_SIGNED_WRAP --emit-llvm-text -o %t.ll

// Constant evaluation must use the declared frontend precision, including
// intermediate operations. These assertions run before backend optimization.
template <typename T> constexpr uniform T add_one(uniform T value) { return value + (uniform T)1; }

static_assert(add_one<float16>(2048.f16) == 2048.f16);
static_assert(add_one<float>(16777216.f) == 16777216.f);
static_assert(add_one<double>(9007199254740992.d) == 9007199254740992.d);
static_assert(add_one<double>(16777216.d) == 16777217.d);
static_assert((uniform float16)1.00048828125d == 1.f16);
static_assert((uniform float)1.000000059604644775390625d == 1.f);
static_assert((uniform uint16)(uniform int16)-1 == 65535);
static_assert((uniform uint32)(uniform int32)-1 == 0xffffffffu);
static_assert((uniform uint64)(uniform int64)-1 == 0xffffffffffffffffull);

// Unsigned arithmetic wraps at the operand width without diagnostics.
template <typename T> constexpr uniform bool wraps(uniform T max) {
    uniform T zero = 0;
    uniform T one = 1;
    return max + one == zero && zero - one == max && max * max == one;
}
static_assert(wraps<uint8>(255));
static_assert(wraps<uint16>(65535));
static_assert(wraps<uint32>(0xffffffffu));
static_assert(wraps<uint64>(0xffffffffffffffffull));

static_assert(((uniform int8)-127 / (uniform int8)2) == (uniform int8)-63);
static_assert(((uniform int16)-32767 % (uniform int16)4) == (uniform int16)-3);
static_assert(((uniform int32)-2147483647 >> 31) == -1);
static_assert((0xffffffffffffffffull >> 63) == 1ull);

#ifdef TEST_SIGNED_WRAP
static_assert(add_one<int8>(127) == (uniform int8)-128);
static_assert(((uniform int16)-32768 - (uniform int16)1) == (uniform int16)32767);
static_assert((1073741824 * 2) == (uniform int32)0x80000000u);
static_assert(add_one<int64>(9223372036854775807ll) == (uniform int64)0x8000000000000000ull);

constexpr uniform int64 negate(uniform int64 value) { return -value; }
static_assert(negate((uniform int64)0x8000000000000000ull) == (uniform int64)0x8000000000000000ull);
#endif

// Unsigned left shifts discard high bits at the frontend operand width. They
// must not inherit C++ integer promotions or produce overflow diagnostics.
static_assert(((uniform uint8)255 << 3) == (uniform uint8)248);
static_assert(((uniform uint16)65535 << 3) == (uniform uint16)65528);
static_assert((0xffffffffu << 3) == 0xfffffff8u);
static_assert((0xffffffffffffffffull << 3) == 0xfffffffffffffff8ull);
static_assert(((uniform int8)-1 << 0) == (uniform int8)-1);
static_assert(((uniform int16)-1 << 0) == (uniform int16)-1);
static_assert(((uniform int32)-1 << 0) == (uniform int32)-1);

// Converting through host double would round these integers to a float tie
// first, and then round the tie in the wrong direction.
static_assert(@llvm.ispc.bitcast((uniform float)0x8000008000000001ull, (uniform uint32)0) == 0x5f000001u);
static_assert(@llvm.ispc.bitcast((uniform float)0x4000004000000001ll, (uniform uint32)0) == 0x5e800001u);
static_assert(@llvm.ispc.bitcast((uniform float)-0x4000004000000001ll, (uniform uint32)0) == 0xde800001u);

constexpr uniform bool aggregate_precision() {
    uniform float16<3> halves = {2048.f16};
    uniform float<3> singles = {16777216.f};
    uniform double<3> doubles = {16777216.d};
    halves[1] = add_one<float16>(halves[0]);
    singles[1] = add_one<float>(singles[0]);
    doubles[1] = add_one<double>(doubles[0]);
    return halves[1] == 2048.f16 && singles[1] == 16777216.f && doubles[1] == 16777217.d &&
           halves[2] == 0.f16 && singles[2] == 0.f && doubles[2] == 0.d;
}
static_assert(aggregate_precision());
