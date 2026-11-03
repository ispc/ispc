// Passing configuration of the nested-offset sum counterexample. Use the
// same valid array indices and runtime oracle as address-gap-nested-sum.ispc.
// The addressing64 path preserves the wide sum.
//
// RUN: %{ispc} %S/address-gap-nested-sum.ispc --pic --target=avx2-i32x8 --addressing=64 -O2 --woff -h %t.h -o %t-a64.o
// RUN: %{cc} -O2 -x c -c %S/address-gap-nested-sum.ispc --include %t.h -o %t.c.o
// RUN: %{cc} %t-a64.o %t.c.o -o %t-a64.bin
// RUN: %t-a64.bin
// Like 3985-1.ispc, execution assumes an AVX2-capable x86-64 host.
// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
