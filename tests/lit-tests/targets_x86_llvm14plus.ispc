// The test checks that target definitions (including all synonyms) are successfully consumed by compiler.

//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512spr-x4
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512spr-x8
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512spr-x16
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512spr-x32
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512spr-x64

// REQUIRES: X86_ENABLED && LLVM_14_0+ && !MACOS_HOST

uniform int j;

int foo(int i) { return i + 1; }
