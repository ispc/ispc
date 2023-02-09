
// Check that --[no]discard-value-names has an effect
// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-llvm-text --discard-value-names -o - | FileCheck %s -check-prefix=CHECK_DISCARD
// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-llvm-text --no-discard-value-names -o - | FileCheck %s -check-prefix=CHECK_NODISCARD

// Check that by default values are generated for LLVM IR dump
// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECK_NODISCARD

// Check that for assembler by default there no names and that we can still see them if commanded so.
// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_DISCARD
// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-asm --no-discard-value-names -o - | FileCheck %s -check-prefix=CHECK_NODISCARD

// CHECK_DISCARD-NOT: allocas
// CHECK_NODISCARD: allocas

extern void goo_for(uniform int);
void foo_for() {
    for (uniform int iter1 = 0; iter1 < 45; iter1++) {
        goo_for(iter1);
    }
}

