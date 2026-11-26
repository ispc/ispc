// The emulated factored gathers of util.m4 read, for the lanes that are off,
// the element of the first lane that is on, and nothing if no lane is on. The
// element at their base may not be readable (here one past the end of the
// array). The kernels and the oracle are in off-lanes-reads.inc.

// RUN: %{ispc} --pic --target=sse2-i32x4 -O2 --woff -h %t-sse2-i32x4.h %S/off-lanes-reads.inc -o %t-sse2-i32x4.o
// RUN: %{cc} -O2 -x c -c %S/off-lanes-reads.inc -o %t-sse2-i32x4.c.o --include %t-sse2-i32x4.h
// RUN: %{cc} %t-sse2-i32x4.o %t-sse2-i32x4.c.o -o %t-sse2-i32x4.bin
// RUN: %t-sse2-i32x4.bin gather_from_end gather_lane2 gather_none_on

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff -h %t-sse4-i32x4.h %S/off-lanes-reads.inc -o %t-sse4-i32x4.o
// RUN: %{cc} -O2 -x c -c %S/off-lanes-reads.inc -o %t-sse4-i32x4.c.o --include %t-sse4-i32x4.h
// RUN: %{cc} %t-sse4-i32x4.o %t-sse4-i32x4.c.o -o %t-sse4-i32x4.bin
// RUN: %t-sse4-i32x4.bin gather_from_end gather_lane2 gather_none_on

// RUN: %{ispc} --pic --target=sse4-i8x16 -O2 --woff -h %t-sse4-i8x16.h %S/off-lanes-reads.inc -o %t-sse4-i8x16.o
// RUN: %{cc} -O2 -x c -c %S/off-lanes-reads.inc -o %t-sse4-i8x16.c.o --include %t-sse4-i8x16.h
// RUN: %{cc} %t-sse4-i8x16.o %t-sse4-i8x16.c.o -o %t-sse4-i8x16.bin
// RUN: %t-sse4-i8x16.bin gather_from_end gather_lane2 gather_none_on

// RUN: %{ispc} --pic --target=avx1-i32x8 -O2 --woff -h %t-avx1-i32x8.h %S/off-lanes-reads.inc -o %t-avx1-i32x8.o
// RUN: %{cc} -O2 -x c -c %S/off-lanes-reads.inc -o %t-avx1-i32x8.c.o --include %t-avx1-i32x8.h
// RUN: %{cc} %t-avx1-i32x8.o %t-avx1-i32x8.c.o -o %t-avx1-i32x8.bin
// RUN: %t-avx1-i32x8.bin gather_from_end gather_lane2 gather_none_on

// RUN: %{ispc} --pic --target=avx2-i8x32 -O2 --woff -h %t-avx2-i8x32.h %S/off-lanes-reads.inc -o %t-avx2-i8x32.o
// RUN: %{cc} -O2 -x c -c %S/off-lanes-reads.inc -o %t-avx2-i8x32.c.o --include %t-avx2-i8x32.h
// RUN: %{cc} %t-avx2-i8x32.o %t-avx2-i8x32.c.o -o %t-avx2-i8x32.bin
// RUN: %t-avx2-i8x32.bin gather_from_end gather_lane2 gather_none_on

// RUN: %{ispc} --pic --target=avx2-i16x16 -O2 --woff -h %t-avx2-i16x16.h %S/off-lanes-reads.inc -o %t-avx2-i16x16.o
// RUN: %{cc} -O2 -x c -c %S/off-lanes-reads.inc -o %t-avx2-i16x16.c.o --include %t-avx2-i16x16.h
// RUN: %{cc} %t-avx2-i16x16.o %t-avx2-i16x16.c.o -o %t-avx2-i16x16.bin
// RUN: %t-avx2-i16x16.bin gather_from_end gather_lane2 gather_none_on

// RUN: %{ispc} --pic --target=avx2-i32x8 -O2 --woff -h %t-avx2-i32x8.h %S/off-lanes-reads.inc -o %t-avx2-i32x8.o
// RUN: %{cc} -O2 -x c -c %S/off-lanes-reads.inc -o %t-avx2-i32x8.c.o --include %t-avx2-i32x8.h
// RUN: %{cc} %t-avx2-i32x8.o %t-avx2-i32x8.c.o -o %t-avx2-i32x8.bin
// RUN: %t-avx2-i32x8.bin gather_from_end gather_lane2 gather_none_on

// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
