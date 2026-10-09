// pdep32/pext32/pdep64/pext64 with dynamic operands use the native BMI2
// instructions on the AVX2, AVX2-VNNI, SKX and ICL targets, at both -O0 and
// -O2. SPR and GNR are covered by pdep-pext-native-4.ispc.
// RUN: %{ispc} %s --target=avx2-i32x4 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x16 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i64x4 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i16x16 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i8x32 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2vnni-i32x4 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2vnni-i32x8 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2vnni-i32x16 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x4 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x8 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x16 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x32 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x64 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512icl-x4 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512icl-x8 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512icl-x16 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512icl-x32 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512icl-x64 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x16-nozmm --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512icl-x16-nozmm --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 -O0 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i16x16 -O0 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x16 -O0 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512icl-x64 -O0 --nowrap --emit-asm -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: pdep_u32___{{.*}}:
// CHECK: pdepl
// CHECK-LABEL: pext_u32___{{.*}}:
// CHECK: pextl
// CHECK-LABEL: pdep_u64___{{.*}}:
// CHECK: pdepq
// CHECK-LABEL: pext_u64___{{.*}}:
// CHECK: pextq
// CHECK-LABEL: pdep_v32___{{.*}}:
// CHECK: pdepl
// CHECK-LABEL: pext_v32___{{.*}}:
// CHECK: pextl
// CHECK-LABEL: pdep_v64___{{.*}}:
// CHECK: pdepq
// CHECK-LABEL: pext_v64___{{.*}}:
// CHECK: pextq

uniform uint32 pdep_u32(uniform uint32 v, uniform uint32 m) { return pdep32(v, m); }
uniform uint32 pext_u32(uniform uint32 v, uniform uint32 m) { return pext32(v, m); }
uniform uint64 pdep_u64(uniform uint64 v, uniform uint64 m) { return pdep64(v, m); }
uniform uint64 pext_u64(uniform uint64 v, uniform uint64 m) { return pext64(v, m); }
uint32 pdep_v32(uint32 v, uint32 m) { return pdep32(v, m); }
uint32 pext_v32(uint32 v, uint32 m) { return pext32(v, m); }
uint64 pdep_v64(uint64 v, uint64 m) { return pdep64(v, m); }
uint64 pext_v64(uint64 v, uint64 m) { return pext64(v, m); }
