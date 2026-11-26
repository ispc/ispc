// A gather whose lanes all read one address becomes a scalar load, which is
// made only if a lane is on: not after a case label that no lane reaches, nor
// in a function called from there with every lane off. The kernels and the
// oracle are in off-lanes-reads.inc.

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff -h %t-sse4-i32x4.h %S/off-lanes-reads.inc -o %t-sse4-i32x4.o
// RUN: %{cc} -O2 -x c -c %S/off-lanes-reads.inc -o %t-sse4-i32x4.c.o --include %t-sse4-i32x4.h
// RUN: %{cc} %t-sse4-i32x4.o %t-sse4-i32x4.c.o -o %t-sse4-i32x4.bin
// RUN: %t-sse4-i32x4.bin broadcast_after_case broadcast_in_callee

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff -h %t-avx2-i32x8.h %S/off-lanes-reads.inc -o %t-avx2-i32x8.o
// RUN: %{cc} -O2 -x c -c %S/off-lanes-reads.inc -o %t-avx2-i32x8.c.o --include %t-avx2-i32x8.h
// RUN: %{cc} %t-avx2-i32x8.o %t-avx2-i32x8.c.o -o %t-avx2-i32x8.bin
// RUN: %t-avx2-i32x8.bin broadcast_after_case broadcast_in_callee

// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
