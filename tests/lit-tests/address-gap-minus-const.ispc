// Known address gap: with 32-bit addressing, InstCombine turns the offset of
// p[idx - 1] (signed idx) into sext((idx << 2) - 4), a 32-bit shift and add
// without nsw, and ImproveMemoryOps distributes the sext over the add as
// sext(idx << 2) - 4, assuming that the add doesn't wrap. Lane k uses
// idx = 2^29 - 3 * k: for lane 0, (idx << 2) - 4 fits in 32 bits but idx << 2
// doesn't, so its address is 4 GB off. The unsigned form is fixed
// (gather_minus_1 in 3985-1.ispc).
//
// XFAIL until the split checks that the add doesn't wrap. The runs below
// have only the cases that fail today; address-gaps-ok.ispc runs the cases
// that pass. See address-gaps.inc for the kernels.

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff -h %t-sse4.h %S/address-gaps.inc -o %t-sse4.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-sse4.c.o --include %t-sse4.h
// RUN: %{cc} %t-sse4.o %t-sse4.c.o -o %t-sse4.bin
// RUN: %t-sse4.bin minus1_gather minus1_scatter

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff -h %t-avx2.h %S/address-gaps.inc -o %t-avx2.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-avx2.c.o --include %t-avx2.h
// RUN: %{cc} %t-avx2.o %t-avx2.c.o -o %t-avx2.bin
// RUN: %t-avx2.bin minus1_scatter

// Like 3985-1.ispc, this assumes that x86-64 hosts support AVX2.
// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
// XFAIL: *
