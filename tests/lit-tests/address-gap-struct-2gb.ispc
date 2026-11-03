// Known address gap: with 32-bit addressing, the byte offset 2^31 of a struct
// member read through a varying pointer is sign extended, so the gather reads
// p - 2 GB instead of p + 2 GB. Through a uniform pointer, and with 64-bit
// addressing, it is right (address-gaps-ok.ispc).
//
// XFAIL until member offsets of 2 GB or more are zero extended. See
// address-gaps.inc for the kernels.

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff -h %t-sse4.h %S/address-gaps.inc -o %t-sse4.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-sse4.c.o --include %t-sse4.h
// RUN: %{cc} %t-sse4.o %t-sse4.c.o -o %t-sse4.bin
// RUN: %t-sse4.bin struct_member_varying

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff -h %t-avx2.h %S/address-gaps.inc -o %t-avx2.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-avx2.c.o --include %t-avx2.h
// RUN: %{cc} %t-avx2.o %t-avx2.c.o -o %t-avx2.bin
// RUN: %t-avx2.bin struct_member_varying

// Like 3985-1.ispc, this assumes that x86-64 hosts support AVX2.
// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
// XFAIL: *
