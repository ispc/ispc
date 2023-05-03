// RUN: %{ispc} %s --target=avx2-i32x8    --nostdlib --emit-asm -o -                         | FileCheck %s -check-prefix=ELIM_FP
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --emit-asm -o -                         | FileCheck %s -check-prefix=ELIM_FP
// RUN: %{ispc} %s --target=avx2-i32x8    --nostdlib --emit-asm --no-omit-frame-pointer -o - | FileCheck %s -check-prefix=NO_ELIM
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --emit-asm --no-omit-frame-pointer -o - | FileCheck %s -check-prefix=NO_ELIM

// REQUIRES: X86_ENABLED

// Test that mechanism behind --no-omit-frame-pointer switch works correctly.

// ELIM_FP-LABEL: foo_leaf
// ELIM_FP-NO: push
// ELIM_FP-NO: pop

// NO_ELIM-LABEL: foo_leaf
// NO_ELIM: push
// NO_ELIM: pop
uniform int foo_leaf() {
  return 10;
}

void foo(uniform int);

// ELIM_FP-LABEL: foo_non_leaf
// ELIM_FP-NO: push
// ELIM_FP-NO: pop

// NO_ELIM-LABEL: foo_non_leaf
// NO_ELIM: push
// NO_ELIM: pop
void foo_non_leaf() {
  foo(0);
}
