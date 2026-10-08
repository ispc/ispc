// Multi-target compilation: targets that share standard-library bitcode
// (SSE4/AVX1/AVX2, and generic/AVX-512) resolve the same pdep/pext builtin
// calls to software on SSE4/AVX1 and to BMI2 on AVX2/AVX-512.
// RUN: %{ispc} %s --target=sse4.2-i32x4,avx1-i32x8,avx2-i32x8,avx512skx-x16 --nowrap --emit-llvm-text -o %t.ll -h %t.h
// RUN: FileCheck %s --input-file=%t_sse4.ll --check-prefix=SW --implicit-check-not=llvm.x86.bmi --implicit-check-not=@__pdep_ --implicit-check-not=@__pext_
// RUN: FileCheck %s --input-file=%t_avx.ll --check-prefix=SW --implicit-check-not=llvm.x86.bmi --implicit-check-not=@__pdep_ --implicit-check-not=@__pext_
// RUN: FileCheck %s --input-file=%t_avx2.ll --check-prefix=NATIVE
// RUN: FileCheck %s --input-file=%t_avx512skx.ll --check-prefix=NATIVE

// REQUIRES: X86_ENABLED

// SW: define {{.*}} @pdep_u32___
// SW: define {{.*}} @pext_u32___
// SW: define {{.*}} @pdep_u64___
// SW: define {{.*}} @pext_u64___
// SW: define {{.*}} @pdep_v32___
// SW: define {{.*}} @pext_v32___
// SW: define {{.*}} @pdep_v64___
// SW: define {{.*}} @pext_v64___

// NATIVE-LABEL: define {{.*}} @pdep_u32___
// NATIVE: call i32 @llvm.x86.bmi.pdep.32(
// NATIVE-LABEL: define {{.*}} @pext_u32___
// NATIVE: call i32 @llvm.x86.bmi.pext.32(
// NATIVE-LABEL: define {{.*}} @pdep_u64___
// NATIVE: call i64 @llvm.x86.bmi.pdep.64(
// NATIVE-LABEL: define {{.*}} @pext_u64___
// NATIVE: call i64 @llvm.x86.bmi.pext.64(
// NATIVE-LABEL: define {{.*}} @pdep_v32___
// NATIVE: call i32 @llvm.x86.bmi.pdep.32(
// NATIVE-LABEL: define {{.*}} @pext_v32___
// NATIVE: call i32 @llvm.x86.bmi.pext.32(
// NATIVE-LABEL: define {{.*}} @pdep_v64___
// NATIVE: call i64 @llvm.x86.bmi.pdep.64(
// NATIVE-LABEL: define {{.*}} @pext_v64___
// NATIVE: call i64 @llvm.x86.bmi.pext.64(

uniform uint32 pdep_u32(uniform uint32 v, uniform uint32 m) { return pdep32(v, m); }
uniform uint32 pext_u32(uniform uint32 v, uniform uint32 m) { return pext32(v, m); }
uniform uint64 pdep_u64(uniform uint64 v, uniform uint64 m) { return pdep64(v, m); }
uniform uint64 pext_u64(uniform uint64 v, uniform uint64 m) { return pext64(v, m); }
uint32 pdep_v32(uint32 v, uint32 m) { return pdep32(v, m); }
uint32 pext_v32(uint32 v, uint32 m) { return pext32(v, m); }
uint64 pdep_v64(uint64 v, uint64 m) { return pdep64(v, m); }
uint64 pext_v64(uint64 v, uint64 m) { return pext64(v, m); }
