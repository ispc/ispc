// The test checks that cpu definitions (including all synonyms) are successfully consumed by compiler.

//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=icelake-client
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=icl

// REQUIRES: LLVM_8_0+

uniform int i;

void foo() {}
