// The test checks that cpu definitions (including all synonyms) are successfully consumed by compiler.

//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=sapphirerapids
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=spr

// REQUIRES: X86_ENABLED && LLVM_14_0+

uniform int i;

void foo() {}
