// The test checks that cpu definitions (including all synonyms) are successfully consumed by compiler.

//; RUN: %{ispc} %s -o %t.o --nostdlib --target=avx512icl-x16 --cpu=znver6

// REQUIRES: X86_ENABLED && LLVM_23_0+

uniform int i;

void foo() {}
