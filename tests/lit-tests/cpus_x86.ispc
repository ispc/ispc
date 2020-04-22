// The test checks that cpu definitions (including all synonyms) are successfully consumed by compiler.

//; RUN: %{ispc} %s -o %t.cpp --emit-c++ --nostdlib --target=generic-4 --cpu=generic
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=x86-64
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=atom
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=bonnell
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=core2
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=penryn
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=slm
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=silvermont
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=corei7
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=nehalem
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=btver2
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=ps4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=corei7-avx
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=sandybridge
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=core-avx-i
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=ivybridge
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=core-avx2
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=haswell
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=broadwell
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=knl
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=skx

uniform int i;

void foo() {}
