// RUN: %{ispc} %s --nostdlib --target=avx2-i32x8 --arch=x86-64 --emit-asm -o - | FileCheck %s -check-prefix=CHECK-NOMODEL-ASM
// RUN: %{ispc} %s --nostdlib --target=avx2-i32x8 --arch=x86-64 --emit-asm --mcmodel=small -o - | FileCheck %s -check-prefix=CHECK-SMALL-ASM
// RUN: %{ispc} %s --nostdlib --target=avx2-i32x8 --arch=x86-64 --emit-asm --mcmodel=large -o - | FileCheck %s -check-prefix=CHECK-LARGE-ASM
// RUN: %{ispc} %s --nostdlib --target=avx2-i32x8 --arch=x86-64 --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECK-NOMODEL-IR
// RUN: %{ispc} %s --nostdlib --target=avx2-i32x8 --arch=x86-64 --emit-llvm-text --mcmodel=small -o - | FileCheck %s -check-prefix=CHECK-SMALL-IR
// RUN: %{ispc} %s --nostdlib --target=avx2-i32x8 --arch=x86-64 --emit-llvm-text --mcmodel=large -o - | FileCheck %s -check-prefix=CHECK-LARGE-IR

// REQUIRES: X86_ENABLED

// CHECK-NOMODEL-ASM: {{lea|array@GOTPCREL}}
// CHECK-SMALL-ASM: {{lea|array@GOTPCREL}}
// CHECK-LARGE-ASM movabsq:
// CHECK-LARGE-ASM-NOT: lea
// CHECK-LARGE-ASM-NOT: array@GOTPCREL

// CHECK-SMALL-IR: !llvm.module.flags = !{{{.*}}}
// CHECK-SMALL-IR: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 1}
// CHECK-LARGE-IR: !llvm.module.flags = !{{{.*}}}
// CHECK-LARGE-IR: !{{[0-9]+}} = !{i32 1, !"Code Model", i32 4}
// CHECK-NOMODEL-IR-NOT: Code Model

uniform int array[100];

uniform int foo(uniform int i) {
  return array[i];
}
