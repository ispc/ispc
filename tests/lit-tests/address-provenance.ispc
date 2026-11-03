// Passing configurations of #3983's integer-built pointer regression.
// A store through origin + integer distance reaches destination: the reload
// must see it. SSE4 exercises both kernels; AVX2 exercises the all-off
// same-block mask and the partial foreach iterations. Check untouched
// bytes/objects too. The failing same-block (n >= 1) and full-foreach cases
// are address-gap-provenance.ispc.

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff -h %t-sse4.h %S/address-provenance.inc -o %t-sse4.o
// RUN: %{cc} -O2 -x c -c %S/address-provenance.inc --include %t-sse4.h -o %t-sse4.c.o
// RUN: %{cc} %t-sse4.o %t-sse4.c.o -o %t-sse4.bin
// RUN: %t-sse4.bin same_block_off
// RUN: %t-sse4.bin same_block
// RUN: %t-sse4.bin across_blocks
// RUN: %t-sse4.bin full_block
// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff -h %t-avx2.h %S/address-provenance.inc -o %t-avx2.o
// RUN: %{cc} -O2 -x c -c %S/address-provenance.inc --include %t-avx2.h -o %t-avx2.c.o
// RUN: %{cc} %t-avx2.o %t-avx2.c.o -o %t-avx2.bin
// RUN: %t-avx2.bin same_block_off
// RUN: %t-avx2.bin across_blocks
// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff --addressing=64 -h %t-avx2-a64.h %S/address-provenance.inc -o %t-avx2-a64.o
// RUN: %{cc} -O2 -x c -c %S/address-provenance.inc --include %t-avx2-a64.h -o %t-avx2-a64.c.o
// RUN: %{cc} %t-avx2-a64.o %t-avx2-a64.c.o -o %t-avx2-a64.bin
// RUN: %t-avx2-a64.bin same_block_off
// RUN: %t-avx2-a64.bin across_blocks

// Like 3985-1.ispc, the AVX2 execution assumes an AVX2-capable x86-64 host.
// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
