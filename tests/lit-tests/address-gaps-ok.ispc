// Runs the cases of address-gaps.inc that pass today, in the configurations
// where they pass: they must keep passing. The cases that fail today are in
// address-gap-lane0.ispc, address-gap-minus-const.ispc and
// address-gap-struct-2gb.ispc (XFAIL); address-gap-wrap.ispc has no case that
// passes today.

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff -h %t-sse4.h %S/address-gaps.inc -o %t-sse4.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-sse4.c.o --include %t-sse4.h
// RUN: %{cc} %t-sse4.o %t-sse4.c.o -o %t-sse4.bin
// RUN: %t-sse4.bin signed_load signed_store struct_member struct_member_store

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff --addressing=64 -h %t-sse4-a64.h %S/address-gaps.inc -o %t-sse4-a64.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-sse4-a64.c.o --include %t-sse4-a64.h
// RUN: %{cc} %t-sse4-a64.o %t-sse4-a64.c.o -o %t-sse4-a64.bin
// RUN: %t-sse4-a64.bin signed_load signed_store signed_load_i8 minus1_gather minus1_scatter struct_member struct_member_store struct_member_varying

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff -h %t-avx2.h %S/address-gaps.inc -o %t-avx2.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-avx2.c.o --include %t-avx2.h
// RUN: %{cc} %t-avx2.o %t-avx2.c.o -o %t-avx2.bin
// RUN: %t-avx2.bin minus1_gather struct_member struct_member_store

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff --addressing=64 -h %t-avx2-a64.h %S/address-gaps.inc -o %t-avx2-a64.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-avx2-a64.c.o --include %t-avx2-a64.h
// RUN: %{cc} %t-avx2-a64.o %t-avx2-a64.c.o -o %t-avx2-a64.bin
// RUN: %t-avx2-a64.bin signed_load signed_store minus1_gather minus1_scatter struct_member struct_member_store struct_member_varying

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff --addressing=64 --wrap-signed-int -h %t-sse4-a64-wrap.h %S/address-gaps.inc -o %t-sse4-a64-wrap.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-sse4-a64-wrap.c.o --include %t-sse4-a64-wrap.h
// RUN: %{cc} %t-sse4-a64-wrap.o %t-sse4-a64-wrap.c.o -o %t-sse4-a64-wrap.bin
// RUN: %t-sse4-a64-wrap.bin signed_load signed_store signed_load_i8

// Like 3985-1.ispc, this assumes that x86-64 hosts support AVX2.
// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
