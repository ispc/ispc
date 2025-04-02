// The test checks that target definitions (including all synonyms) are successfully consumed by compiler.

//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2-x4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2-x8
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2-x16
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2-x32
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2-x64

// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_20_0+

uniform int j;

int foo(int i) { return i + 1; }
