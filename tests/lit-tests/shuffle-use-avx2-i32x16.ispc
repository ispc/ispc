// Test shuffle asm codegen inside foreach loop with reads/writes and computation.
// Ensures no old inefficient vpextrd/vpinsrd instructions are generated.

// RUN: %{ispc} %s --target=avx2-i32x16 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X16

// REQUIRES: X86_ENABLED

// CHECK-X16-LABEL: shuffle1_i32___un_3C_uni_3E_un_3C_uni_3E_vyiuni:
// CHECK-X16-DAG: vpermd
// CHECK-X16-NOT: vpextrd
// CHECK-X16-NOT: vpinsrd
void shuffle1_i32(uniform int32 a[], uniform int32 ret[], int perm, uniform int count) {
    assume(count % 64 == 0);
    foreach (i = 0...count) {
        ret[i] = shuffle(a[i], programIndex + perm);
    }
}

// CHECK-X16-LABEL: shuffle1_f32___un_3C_unf_3E_un_3C_unf_3E_vyiuni:
// CHECK-X16-DAG: vpermd
// CHECK-X16-NOT: vpextrd
// CHECK-X16-NOT: vinsertps
void shuffle1_f32(uniform float a[], uniform float ret[], int perm, uniform int count) {
    assume(count % 64 == 0);
    foreach (i = 0...count) {
        ret[i] = shuffle(a[i], programIndex + perm);
    }
}

// CHECK-X16-LABEL: shuffle2_i32___un_3C_uni_3E_un_3C_uni_3E_un_3C_uni_3E_vyiuni:
// CHECK-X16-DAG: vpermd
// CHECK-X16-NOT: vpextrd
// CHECK-X16-NOT: vpinsrd
void shuffle2_i32(uniform int32 a[], uniform int32 b[], uniform int32 ret[], int perm, uniform int count) {
    assume(count % 64 == 0);
    foreach (i = 0...count) {
        ret[i] = shuffle(a[i], b[i], programIndex + perm);
    }
}

// CHECK-X16-LABEL: shuffle2_f32___un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_vyiuni:
// CHECK-X16-DAG: vpermd
// CHECK-X16-NOT: vpextrd
// CHECK-X16-NOT: vinsertps
void shuffle2_f32(uniform float a[], uniform float b[], uniform float ret[], int perm, uniform int count) {
    assume(count % 64 == 0);
    foreach (i = 0...count) {
        ret[i] = shuffle(a[i], b[i], programIndex + perm);
    }
}
