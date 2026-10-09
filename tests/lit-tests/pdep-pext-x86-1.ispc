// On 32-bit x86, the 32-bit pdep/pext builtins use BMI2, while the 64-bit
// builtins resolve to the generic software definitions: the 64-bit BMI2
// instructions require 64-bit mode.
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86 -O0 --nowrap --debug-phase=10:10 -o %t.o | FileCheck %s --implicit-check-not="{{llvm.x86.bmi.pdep.64|llvm.pdep.i64}}" --implicit-check-not="{{llvm.x86.bmi.pext.64|llvm.pext.i64}}"
// RUN: %{ispc} %s --target=avx2-i16x16 --arch=x86 -O0 --nowrap --debug-phase=10:10 -o %t.o | FileCheck %s --implicit-check-not="{{llvm.x86.bmi.pdep.64|llvm.pdep.i64}}" --implicit-check-not="{{llvm.x86.bmi.pext.64|llvm.pext.i64}}"
// RUN: %{ispc} %s --target=avx512skx-x16 --arch=x86 -O0 --nowrap --debug-phase=10:10 -o %t.o | FileCheck %s --implicit-check-not="{{llvm.x86.bmi.pdep.64|llvm.pdep.i64}}" --implicit-check-not="{{llvm.x86.bmi.pext.64|llvm.pext.i64}}"
// RUN: %{ispc} %s --target=avx512icl-x8 --arch=x86 -O0 --nowrap --debug-phase=10:10 -o %t.o | FileCheck %s --implicit-check-not="{{llvm.x86.bmi.pdep.64|llvm.pdep.i64}}" --implicit-check-not="{{llvm.x86.bmi.pext.64|llvm.pext.i64}}"
// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86 -O2 --nowrap --debug-phase=10:10 -o %t.o | FileCheck %s --implicit-check-not="{{llvm.x86.bmi.pdep.64|llvm.pdep.i64}}" --implicit-check-not="{{llvm.x86.bmi.pext.64|llvm.pext.i64}}"
// RUN: %{ispc} %s --target=avx2-i16x16 --arch=x86 -O2 --nowrap --debug-phase=10:10 -o %t.o | FileCheck %s --implicit-check-not="{{llvm.x86.bmi.pdep.64|llvm.pdep.i64}}" --implicit-check-not="{{llvm.x86.bmi.pext.64|llvm.pext.i64}}"
// RUN: %{ispc} %s --target=avx512skx-x16 --arch=x86 -O2 --nowrap --debug-phase=10:10 -o %t.o | FileCheck %s --implicit-check-not="{{llvm.x86.bmi.pdep.64|llvm.pdep.i64}}" --implicit-check-not="{{llvm.x86.bmi.pext.64|llvm.pext.i64}}"
// RUN: %{ispc} %s --target=avx512icl-x8 --arch=x86 -O2 --nowrap --debug-phase=10:10 -o %t.o | FileCheck %s --implicit-check-not="{{llvm.x86.bmi.pdep.64|llvm.pdep.i64}}" --implicit-check-not="{{llvm.x86.bmi.pext.64|llvm.pext.i64}}"

// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK-DAG: define internal {{.*}} @__pdep_uniform_i32(
// CHECK-DAG: define internal {{.*}} @__pdep_uniform_i64(
// CHECK-DAG: define internal {{.*}} @__pdep_varying_i32(
// CHECK-DAG: define internal {{.*}} @__pdep_varying_i64(
// CHECK-DAG: define internal {{.*}} @__pext_uniform_i32(
// CHECK-DAG: define internal {{.*}} @__pext_uniform_i64(
// CHECK-DAG: define internal {{.*}} @__pext_varying_i32(
// CHECK-DAG: define internal {{.*}} @__pext_varying_i64(
// CHECK-DAG: call i32 @llvm.{{x86.bmi.pdep.32|pdep.i32}}(
// CHECK-DAG: call i32 @llvm.{{x86.bmi.pext.32|pext.i32}}(

uniform uint32 pdep_u32(uniform uint32 v, uniform uint32 m) { return pdep32(v, m); }
uniform uint32 pext_u32(uniform uint32 v, uniform uint32 m) { return pext32(v, m); }
uniform uint64 pdep_u64(uniform uint64 v, uniform uint64 m) { return pdep64(v, m); }
uniform uint64 pext_u64(uniform uint64 v, uniform uint64 m) { return pext64(v, m); }
uint32 pdep_v32(uint32 v, uint32 m) { return pdep32(v, m); }
uint32 pext_v32(uint32 v, uint32 m) { return pext32(v, m); }
uint64 pdep_v64(uint64 v, uint64 m) { return pdep64(v, m); }
uint64 pext_v64(uint64 v, uint64 m) { return pext64(v, m); }
