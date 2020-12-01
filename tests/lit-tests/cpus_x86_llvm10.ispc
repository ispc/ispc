// The test checks that cpu definitions (including all synonyms) are successfully consumed by compiler.

//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=tigerlake
//; RUN: %{ispc} %s -o %t.o --nostdlib --target=sse2-i32x4 --cpu=tgl

// REQUIRES: X86_ENABLED
// REQUIRES: LLVM_10_0+

uniform int i;

void foo() {}
