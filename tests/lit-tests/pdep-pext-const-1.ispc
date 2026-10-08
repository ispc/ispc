// pdep/pext with compile-time constant masks: the software targets and the
// varying versions on BMI2 targets use straight-line shift/AND/OR code without
// loops, branches, or per-instance BMI2 instructions. The uniform versions on
// BMI2 targets keep the native instruction, also for 32-bit operations on
// 32-bit x86.
// RUN: %{ispc} %s --target=sse2-i32x4 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK --implicit-check-not=llvm.x86.bmi
// RUN: %{ispc} %s --target=sse4.2-i32x4 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK --implicit-check-not=llvm.x86.bmi
// RUN: %{ispc} %s --target=avx1-i32x8 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK --implicit-check-not=llvm.x86.bmi
// RUN: %{ispc} %s --target=generic-i1x16 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK --implicit-check-not=llvm.x86.bmi
// RUN: %{ispc} %s --target=generic-i32x8 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK --implicit-check-not=llvm.x86.bmi
// RUN: %{ispc} %s --target=avx2-i32x8 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,NATIVE
// RUN: %{ispc} %s --target=avx2-i16x16 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,NATIVE
// RUN: %{ispc} %s --target=avx2-i64x4 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,NATIVE
// RUN: %{ispc} %s --target=avx512skx-x16 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,NATIVE
// RUN: %{ispc} %s --target=avx512skx-x64 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,NATIVE
// RUN: %{ispc} %s --target=avx512icl-x8 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,NATIVE
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,X86
// RUN: %{ispc} %s --target=avx512skx-x16 --arch=x86 -O2 --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefixes=CHECK,X86

// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK-LABEL: define {{.*}} @pdep_u32___
// NATIVE: call i32 @llvm.x86.bmi.pdep.32(
// X86: call i32 @llvm.x86.bmi.pdep.32(
// CHECK-NOT: {{br |phi |llvm.x86.bmi|llvm.cttz}}
// CHECK: ret
// CHECK-LABEL: define {{.*}} @pext_u32___
// NATIVE: call i32 @llvm.x86.bmi.pext.32(
// X86: call i32 @llvm.x86.bmi.pext.32(
// CHECK-NOT: {{br |phi |llvm.x86.bmi|llvm.cttz}}
// CHECK: ret
// CHECK-LABEL: define {{.*}} @pdep_u64___
// NATIVE: call i64 @llvm.x86.bmi.pdep.64(
// CHECK-NOT: {{br |phi |llvm.x86.bmi|llvm.cttz}}
// CHECK: ret
// CHECK-LABEL: define {{.*}} @pext_u64___
// NATIVE: call i64 @llvm.x86.bmi.pext.64(
// CHECK-NOT: {{br |phi |llvm.x86.bmi|llvm.cttz}}
// CHECK: ret
// CHECK-LABEL: define {{.*}} @pdep_v32___
// CHECK-NOT: {{br |phi |llvm.x86.bmi|llvm.cttz}}
// CHECK: ret
// CHECK-LABEL: define {{.*}} @pext_v32___
// CHECK-NOT: {{br |phi |llvm.x86.bmi|llvm.cttz}}
// CHECK: ret
// CHECK-LABEL: define {{.*}} @pdep_v64___
// CHECK-NOT: {{br |phi |llvm.x86.bmi|llvm.cttz}}
// CHECK: ret
// CHECK-LABEL: define {{.*}} @pext_v64___
// CHECK-NOT: {{br |phi |llvm.x86.bmi|llvm.cttz}}
// CHECK: ret

uniform uint32 pdep_u32(uniform uint32 x) { return pdep32(x, 0x55555555u); }
uniform uint32 pext_u32(uniform uint32 x) { return pext32(x, 0x00FF00F0u); }
uniform uint64 pdep_u64(uniform uint64 x) { return pdep64(x, 0x1249249249249249ull); }
uniform uint64 pext_u64(uniform uint64 x) { return pext64(x, 0xAAAAAAAAAAAAAAAAull); }
uint32 pdep_v32(uint32 x) { return pdep32(x, 0x55555555u); }
uint32 pext_v32(uint32 x) { return pext32(x, 0x00FF00F0u); }
uint64 pdep_v64(uint64 x) { return pdep64(x, 0x1249249249249249ull); }
uint64 pext_v64(uint64 x) { return pext64(x, 0xAAAAAAAAAAAAAAAAull); }
