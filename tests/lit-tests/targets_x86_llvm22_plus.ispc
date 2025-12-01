// The test checks that target definitions (including all synonyms) are successfully consumed by compiler.

//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2nvl-x4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2nvl-x8
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2nvl-x16
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2nvl-x32
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2nvl-x64

// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_22_0+

uniform int j;

int foo(int i) { return i + 1; }
