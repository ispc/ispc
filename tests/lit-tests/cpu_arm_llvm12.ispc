// The test checks that cpu definitions (including all synonyms) are successfully consumed by compiler.

//; RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --cpu=apple-a14

// REQUIRES: ARM_ENABLED && !WINDOWS_ENABLED
// REQUIRES: LLVM_12_0+

uniform int i;

void foo() {}
