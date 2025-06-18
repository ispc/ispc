// Test rotate asm codegen for avx512 targets

// This checks that the default builtin implementation is not used for targets
// where it is possible to use any variant/combination of permutation
// instructions. To do that we just check that no vector stores to stack are
// generated.

// RUN: %{ispc} %s --target=avx512skx-x4  --x86-asm-syntax=intel --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X4
// RUN: %{ispc} %s --target=avx512skx-x8  --x86-asm-syntax=intel --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X8
// RUN: %{ispc} %s --target=avx512skx-x16 --x86-asm-syntax=intel --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X16
// RUN: %{ispc} %s --target=avx512skx-x32 --x86-asm-syntax=intel --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X32
// RUN: %{ispc} %s --target=avx512skx-x64 --x86-asm-syntax=intel --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X64
// RUN: %{ispc} %s --target=avx512icl-x4  --x86-asm-syntax=intel --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X4
// RUN: %{ispc} %s --target=avx512icl-x8  --x86-asm-syntax=intel --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X8
// RUN: %{ispc} %s --target=avx512icl-x16 --x86-asm-syntax=intel --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X16
// RUN: %{ispc} %s --target=avx512icl-x32 --x86-asm-syntax=intel --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X32
// RUN: %{ispc} %s --target=avx512icl-x64 --x86-asm-syntax=intel --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X64

// REQUIRES: X86_ENABLED

// CHECK-X4-LABEL: rotate_i8___vytuni:
// CHECK-X4-DAG: vpshufb
// CHECK-X4-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], xmm{{[0-9]+}}
// CHECK-X8-LABEL: rotate_i8___vytuni:
// CHECK-X8-DAG: vpshufb
// CHECK-X8-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], xmm{{[0-9]+}}
// CHECK-X16-LABEL: rotate_i8___vytuni:
//   vpshufb for skx and vpermb for icl
// CHECK-X16-DAG: {{vpshufb|vpermb}}
// CHECK-X16-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], xmm{{[0-9]+}}
// CHECK-X32-LABEL: rotate_i8___vytuni:
//   vpermi2w for skx and vpermb for icl
// CHECK-X32-DAG: {{vpermw|vpermb}}
// CHECK-X32-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], ymm{{[0-9]+}}
// CHECK-X64-LABEL: rotate_i8___vytuni:
//   vpermi2w for skx and vpermb for icl
// CHECK-X64-DAG: {{vpermi2w|vpermb}}
// CHECK-X64-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
int8 rotate_i8(int8 x, uniform int32 r) {
    return rotate(x, r);
}

// CHECK-X4-LABEL: rotate_i16___vysuni:
// CHECK-X4-DAG: vpermw
// CHECK-X4-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], xmm{{[0-9]+}}
// CHECK-X8-LABEL: rotate_i16___vysuni:
// CHECK-X8-DAG: vpermw
// CHECK-X8-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], xmm{{[0-9]+}}
// CHECK-X16-LABEL: rotate_i16___vysuni:
// CHECK-X16-DAG: vpermw
// CHECK-X16-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], ymm{{[0-9]+}}
// CHECK-X32-LABEL: rotate_i16___vysuni:
// CHECK-X32-DAG: vpermw
// CHECK-X32-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X64-LABEL: rotate_i16___vysuni:
// CHECK-X64-DAG: vpermi2w
// CHECK-X64-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
int16 rotate_i16(int16 x, uniform int32 r) {
    return rotate(x, r);
}

// CHECK-X4-LABEL: rotate_i32___vyiuni:
// CHECK-X4-DAG: vpermilps
// CHECK-X4-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], xmm{{[0-9]+}}
// CHECK-X8-LABEL: rotate_i32___vyiuni:
// CHECK-X8-DAG: vpermps
// CHECK-X8-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], ymm{{[0-9]+}}
// CHECK-X16-LABEL: rotate_i32___vyiuni:
// CHECK-X16-DAG: vpermps
// CHECK-X16-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X32-LABEL: rotate_i32___vyiuni:
// CHECK-X32-DAG: vpermi2d
// CHECK-X32-DAG: vpermi2d
// CHECK-X32-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X64-LABEL: rotate_i32___vyiuni:
// CHECK-X64: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X64-NOT: vpgatherdd
int32 rotate_i32(int32 x, uniform int32 r) {
    return rotate(x, r);
}

// CHECK-X4-LABEL: rotate_i64___vyIuni:
// CHECK-X4-DAG: vpermq
// CHECK-X4-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], ymm{{[0-9]+}}
// CHECK-X8-LABEL: rotate_i64___vyIuni:
// CHECK-X8-DAG: vpermi2q
// CHECK-X8-DAG: vpermi2q
// CHECK-X8-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], ymm{{[0-9]+}}
// CHECK-X16-LABEL: rotate_i64___vyIuni:
// CHECK-X16-DAG: vpermi2q
// CHECK-X16-DAG: vpermi2q
// CHECK-X16-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X32-LABEL: rotate_i64___vyIuni:
// CHECK-X32: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X32-NOT: vpgatherdq
// CHECK-X64-LABEL: rotate_i64___vyIuni:
// CHECK-X64: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X64-NOT: vpgatherdq
int64 rotate_i64(int64 x, uniform int32 r) {
    return rotate(x, r);
}

// CHECK-X4-LABEL: rotate_half___vyhuni:
// CHECK-X4-DAG: vpermw
// CHECK-X4-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], xmm{{[0-9]+}}
// CHECK-X8-LABEL: rotate_half___vyhuni:
// CHECK-X8-DAG: vpermw
// CHECK-X8-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], xmm{{[0-9]+}}
// CHECK-X16-LABEL: rotate_half___vyhuni:
// CHECK-X16-DAG: vpermw
// CHECK-X16-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], ymm{{[0-9]+}}
// CHECK-X32-LABEL: rotate_half___vyhuni:
// CHECK-X32-DAG: vpermw
// CHECK-X32-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X64-LABEL: rotate_half___vyhuni:
// CHECK-X64-DAG: vpermi2w
// CHECK-X64-DAG: vpermi2w
// CHECK-X64-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
float16 rotate_half(float16 x, uniform int32 r) {
    return rotate(x, r);
}

// CHECK-X4-LABEL: rotate_float___vyfuni:
// CHECK-X4-DAG: vpermilps
// CHECK-X4-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], xmm{{[0-9]+}}
// CHECK-X8-LABEL: rotate_float___vyfuni:
// CHECK-X8-DAG: vpermps
// CHECK-X8-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], ymm{{[0-9]+}}
// CHECK-X16-LABEL: rotate_float___vyfuni:
// CHECK-X16-DAG: vpermps
// CHECK-X16-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X32-LABEL: rotate_float___vyfuni:
// CHECK-X32-DAG: vpermi2ps
// CHECK-X32-DAG: vpermi2ps
// CHECK-X32-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X64-LABEL: rotate_float___vyfuni:
// CHECK-X64: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X64-NOT: vpgatherdd
float rotate_float(float x, uniform int32 r) {
    return rotate(x, r);
}

// CHECK-X4-LABEL: rotate_double___vyduni:
// CHECK-X4-DAG: vpermq
// CHECK-X4-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], ymm{{[0-9]+}}
// CHECK-X8-LABEL: rotate_double___vyduni:
// CHECK-X8-DAG: vpermi2pd
// CHECK-X8-DAG: vpermi2pd
// CHECK-X8-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], ymm{{[0-9]+}}
// CHECK-X16-LABEL: rotate_double___vyduni:
// CHECK-X16-DAG: vpermi2pd
// CHECK-X16-DAG: vpermi2pd
// CHECK-X16-NOT: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X32-LABEL: rotate_double___vyduni:
// CHECK-X32: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X32-NOT: vpgatherdq
// CHECK-X64-LABEL: rotate_double___vyduni:
// CHECK-X64: vmov{{.*}} {{.*}}word ptr [rsp{{.*}}], zmm{{[0-9]+}}
// CHECK-X64-NOT: vpgatherdq
double rotate_double(double x, uniform int32 r) {
    return rotate(x, r);
}
