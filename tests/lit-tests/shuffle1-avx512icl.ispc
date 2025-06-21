// Test shuffle asm codegen for i8 for icl targets
// ICL targets can use the new vpermb instruction (AVX-512 VBMI) for efficient
// i8 shuffles on larger vectors (x16+).

// RUN: %{ispc} %s --target=avx512icl-x4 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X4
// RUN: %{ispc} %s --target=avx512icl-x8 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X8
// RUN: %{ispc} %s --target=avx512icl-x16 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X16
// RUN: %{ispc} %s --target=avx512icl-x32 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X32
// RUN: %{ispc} %s --target=avx512icl-x64 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X64

// REQUIRES: X86_ENABLED

// CHECK-X4-LABEL: shuffle_i8___vytvyi:
// CHECK-X4-DAG: vpshufb
// CHECK-X4-NOT: vpextrd
// CHECK-X4-NOT: vpinsrb
// CHECK-X8-LABEL: shuffle_i8___vytvyi:
// CHECK-X8-DAG: vpshufb
// CHECK-X8-NOT: vpextrd
// CHECK-X8-NOT: vpinsrb
// CHECK-X16-LABEL: shuffle_i8___vytvyi:
// CHECK-X16-DAG: vpermb
// CHECK-X16-NOT: vpextrd
// CHECK-X16-NOT: vpinsrb
// CHECK-X32-LABEL: shuffle_i8___vytvyi:
// CHECK-X32-DAG: vpermb
// CHECK-X32-NOT: vpextrd
// CHECK-X32-NOT: vpinsrb
// CHECK-X64-LABEL: shuffle_i8___vytvyi:
// CHECK-X64-DAG: vpermb
// CHECK-X64-NOT: vpextrd
// CHECK-X64-NOT: vpinsrb
int8 shuffle_i8(int8 x, int32 perm) {
    return shuffle(x, perm);
}
