// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Check a library-style range contract before narrowing an intrinsic immediate.
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap -O0 --off-phase=all --emit-llvm-text -o - | FileCheck %s
// RUN: not %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap -DBAD_SCALE=0 -o %t.o 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: not %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap -DBAD_SCALE=3 --opt=disable-assertions -o %t.o 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: not %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --nowrap -DBAD_SCALE=257 -o %t.o 2>&1 | FileCheck %s --check-prefix=ERROR
// REQUIRES: X86_ENABLED

template <uint64 Scale>
varying int gather_scaled(uniform int8 *uniform base, varying int offsets, varying int mask) {
    static_assert(Scale == 1 || Scale == 2 || Scale == 4 || Scale == 8,
                  "Gather scale must be one of 1, 2, 4, or 8.");
    return @llvm.x86.avx2.gather.d.d.256((varying int)0, base, offsets, mask, (uniform int8)Scale);
}

constexpr uniform uint64 word_scale(uniform uint64 width) { return width * 2; }

template <uint64 Width>
varying int gather_words(uniform int8 *uniform base, varying int offsets, varying int mask) {
    constexpr uniform uint64 scale = word_scale(Width);
    return gather_scaled<scale>(base, offsets, mask);
}

varying int test_scales(uniform int8 *uniform base, varying int offsets, varying int mask) {
#ifdef BAD_SCALE
    return gather_scaled<BAD_SCALE>(base, offsets, mask);
#else
    return gather_scaled<1>(base, offsets, mask) + gather_words<1>(base, offsets, mask) +
           gather_words<2>(base, offsets, mask) + gather_words<4>(base, offsets, mask);
#endif
}

// CHECK-DAG: call <8 x i32> @llvm.x86.avx2.gather.d.d.256({{.*}}, i8 1)
// CHECK-DAG: call <8 x i32> @llvm.x86.avx2.gather.d.d.256({{.*}}, i8 2)
// CHECK-DAG: call <8 x i32> @llvm.x86.avx2.gather.d.d.256({{.*}}, i8 4)
// CHECK-DAG: call <8 x i32> @llvm.x86.avx2.gather.d.d.256({{.*}}, i8 8)
// ERROR: Error: static assertion failed: Gather scale must be one of 1, 2, 4, or 8.
