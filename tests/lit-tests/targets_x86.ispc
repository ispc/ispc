// The test checks that target definitions (including all synonyms) are successfully consumed by compiler.

//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x8
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse4-i8x16
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse4-i16x8
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse4-i32x4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse4-i32x8
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse4.1-i8x16
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse4.1-i16x8
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse4.1-i32x4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse4.1-i32x8
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx1-i32x4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx1-i32x8
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx1-i32x16
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx1-i64x4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx2-i8x32
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx2-i16x16
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx2-i32x4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx2-i32x8
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx2-i32x16
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx2-i64x4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512knl-x16
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512knl-i32x16
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512skx-x4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512skx-i32x4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512skx-x8
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512skx-i32x8
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512skx-x16
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512skx-i32x16
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512skx-x32
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512skx-i16x32
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512skx-x64
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512skx-i8x64

// REQUIRES: X86_ENABLED

uniform int j;

int foo(int i) { return i + 1; }
