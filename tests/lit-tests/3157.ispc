// RUN: %{ispc} --target=avx512icl-x4 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx512icl-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx512icl-x16 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx512icl-x32 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx512icl-x64 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s

// RUN: %{ispc} --target=avx512spr-x4 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx512spr-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx512spr-x16 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx512spr-x32 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx512spr-x64 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s

// RUN: %{ispc} --target=avx512gnr-x4 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx512gnr-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx512gnr-x16 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx512gnr-x32 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx512gnr-x64 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s

// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK-LABEL: clz___vyi
// CHECK: vplzcntd
// CHECK-NOT: vpextrd
// CHECK-NOT: vpinsrd
// CHECK-NOT: lzcnt   r

// CHECK-LABEL: clz___vyI
// CHECK: vplzcntq
// CHECK-NOT: vextracti128
// CHECK-NOT: vinserti128
// CHECK-NOT: lzcnt   r

// CHECK-LABEL: ctz___vyi
// CHECK: vpopcntd
// CHECK-NOT: vpextrd
// CHECK-NOT: vpinsrd
// CHECK-NOT: tzcnt   r

// CHECK-LABEL: ctz___vyI
// CHECK: vpopcntq
// CHECK-NOT: vextracti128
// CHECK-NOT: vinserti128
// CHECK-NOT: tzcnt   r

int32 clz(int32 v) {
    return count_leading_zeros(v);
}
int64 clz(int64 v) {
    return count_leading_zeros(v);
}
int32 ctz(int32 v) {
    return count_trailing_zeros(v);
}
int64 ctz(int64 v) {
    return count_trailing_zeros(v);
}

