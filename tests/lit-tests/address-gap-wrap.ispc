// Known address gap: offsets that are linear except for a wrap at 2^32 count
// as linear, so an access with all lanes on becomes one vector load or store
// although the index wraps from 2^32 - 1 to 0 between two lanes
// (p8[b + programIndex] with an unsigned b = 2^32 - 2), or from INT32_MAX to
// INT32_MIN with --wrap-signed-int. No single address serves all lanes. The
// masked forms, whose lane 0 is off before the wrap, are covered by
// 3985-1.ispc.
//
// XFAIL until such offsets are proved not to wrap or are lowered per lane.
// All runs fail today. See address-gaps.inc for the kernels.

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff -h %t-sse4.h %S/address-gaps.inc -o %t-sse4.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-sse4.c.o --include %t-sse4.h
// RUN: %{cc} %t-sse4.o %t-sse4.c.o -o %t-sse4.bin
// RUN: %t-sse4.bin wrap_load_i8 wrap_store_i8 foreach_wrap_load_i8

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff --addressing=64 -h %t-sse4-a64.h %S/address-gaps.inc -o %t-sse4-a64.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-sse4-a64.c.o --include %t-sse4-a64.h
// RUN: %{cc} %t-sse4-a64.o %t-sse4-a64.c.o -o %t-sse4-a64.bin
// RUN: %t-sse4-a64.bin wrap_load_i8 wrap_store_i8 foreach_wrap_load_i8

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff -h %t-avx2.h %S/address-gaps.inc -o %t-avx2.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-avx2.c.o --include %t-avx2.h
// RUN: %{cc} %t-avx2.o %t-avx2.c.o -o %t-avx2.bin
// RUN: %t-avx2.bin wrap_load_i8 wrap_store_i8 foreach_wrap_load_i8

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff --addressing=64 -h %t-avx2-a64.h %S/address-gaps.inc -o %t-avx2-a64.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-avx2-a64.c.o --include %t-avx2-a64.h
// RUN: %{cc} %t-avx2-a64.o %t-avx2-a64.c.o -o %t-avx2-a64.bin
// RUN: %t-avx2-a64.bin wrap_load_i8 wrap_store_i8 foreach_wrap_load_i8

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff --wrap-signed-int -h %t-sse4-wrap.h %S/address-gaps.inc -o %t-sse4-wrap.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-sse4-wrap.c.o --include %t-sse4-wrap.h
// RUN: %{cc} %t-sse4-wrap.o %t-sse4-wrap.c.o -o %t-sse4-wrap.bin
// RUN: %t-sse4-wrap.bin signed_wrap_load_i8

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff --addressing=64 --wrap-signed-int -h %t-avx2-a64-wrap.h %S/address-gaps.inc -o %t-avx2-a64-wrap.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-avx2-a64-wrap.c.o --include %t-avx2-a64-wrap.h
// RUN: %{cc} %t-avx2-a64-wrap.o %t-avx2-a64-wrap.c.o -o %t-avx2-a64-wrap.bin
// RUN: %t-avx2-a64-wrap.bin signed_wrap_load_i8

// Like 3985-1.ispc, this assumes that x86-64 hosts support AVX2.
// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
// XFAIL: *
