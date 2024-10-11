// The test checks that cpu definitions (including all synonyms) are successfully consumed by compiler.

//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=arl
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=arrowlake
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=lnl
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=lunarlake

// REQUIRES: X86_ENABLED && LLVM_18_0+

uniform int i;

void foo() {}