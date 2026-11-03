// Known address gap: a masked access whose lanes that are on access
// contiguous memory becomes a vector load or store, and ImproveMemoryOps takes
// its address from lane 0, also when lane 0 is off. For a signed index whose
// byte offset in lane 0 doesn't fit in 32 bits (the offsets of the lanes that
// are on do), that address is 4 GB off or poison. Only unsigned offsets that
// may wrap use the first lane that is on (3985-1.ispc).
//
// XFAIL until the address of such an access comes from a lane that is on.
// The runs below have only the cases that fail today; address-gaps-ok.ispc
// runs the cases that pass. See address-gaps.inc for the kernels.

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff -h %t-sse4.h %S/address-gaps.inc -o %t-sse4.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-sse4.c.o --include %t-sse4.h
// RUN: %{cc} %t-sse4.o %t-sse4.c.o -o %t-sse4.bin
// RUN: %t-sse4.bin signed_load_i8

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff --wrap-signed-int -h %t-sse4-wrap.h %S/address-gaps.inc -o %t-sse4-wrap.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-sse4-wrap.c.o --include %t-sse4-wrap.h
// RUN: %{cc} %t-sse4-wrap.o %t-sse4-wrap.c.o -o %t-sse4-wrap.bin
// RUN: %t-sse4-wrap.bin signed_load signed_store signed_load_i8

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff -h %t-avx2.h %S/address-gaps.inc -o %t-avx2.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-avx2.c.o --include %t-avx2.h
// RUN: %{cc} %t-avx2.o %t-avx2.c.o -o %t-avx2.bin
// RUN: %t-avx2.bin signed_load signed_store signed_load_i8

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff --addressing=64 -h %t-avx2-a64.h %S/address-gaps.inc -o %t-avx2-a64.o
// RUN: %{cc} -O2 -x c -c %S/address-gaps.inc -o %t-avx2-a64.c.o --include %t-avx2-a64.h
// RUN: %{cc} %t-avx2-a64.o %t-avx2-a64.c.o -o %t-avx2-a64.bin
// RUN: %t-avx2-a64.bin signed_load_i8

// Like 3985-1.ispc, this assumes that x86-64 hosts support AVX2.
// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
// XFAIL: *
