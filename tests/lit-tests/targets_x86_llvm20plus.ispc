// The test checks that target definitions (including all synonyms) are successfully consumed by compiler.

//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2dmr-x4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2dmr-x8
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2dmr-x16
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2dmr-x32
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx10.2dmr-x64

// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_20_0+

uniform int j;

int foo(int i) { return i + 1; }
