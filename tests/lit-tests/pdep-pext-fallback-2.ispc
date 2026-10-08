// pdep/pext compile to the generic software definitions on ARM NEON targets.
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --nowrap --emit-llvm-text -o - | FileCheck %s --implicit-check-not=llvm.x86.bmi --implicit-check-not=@__pdep_ --implicit-check-not=@__pext_
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i8x16 --nowrap --emit-llvm-text -o - | FileCheck %s --implicit-check-not=llvm.x86.bmi --implicit-check-not=@__pdep_ --implicit-check-not=@__pext_
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x8 --nowrap --emit-llvm-text -o - | FileCheck %s --implicit-check-not=llvm.x86.bmi --implicit-check-not=@__pdep_ --implicit-check-not=@__pext_
// RUN: %{ispc} %s --arch=arm --target=neon-i32x4 --nowrap --emit-llvm-text -o - | FileCheck %s --implicit-check-not=llvm.x86.bmi --implicit-check-not=@__pdep_ --implicit-check-not=@__pext_

// REQUIRES: ARM_ENABLED

// CHECK: define {{.*}} @pdep_u32___
// CHECK: define {{.*}} @pext_u32___
// CHECK: define {{.*}} @pdep_u64___
// CHECK: define {{.*}} @pext_u64___
// CHECK: define {{.*}} @pdep_v32___
// CHECK: define {{.*}} @pext_v32___
// CHECK: define {{.*}} @pdep_v64___
// CHECK: define {{.*}} @pext_v64___

uniform uint32 pdep_u32(uniform uint32 v, uniform uint32 m) { return pdep32(v, m); }
uniform uint32 pext_u32(uniform uint32 v, uniform uint32 m) { return pext32(v, m); }
uniform uint64 pdep_u64(uniform uint64 v, uniform uint64 m) { return pdep64(v, m); }
uniform uint64 pext_u64(uniform uint64 v, uniform uint64 m) { return pext64(v, m); }
uint32 pdep_v32(uint32 v, uint32 m) { return pdep32(v, m); }
uint32 pext_v32(uint32 v, uint32 m) { return pext32(v, m); }
uint64 pdep_v64(uint64 v, uint64 m) { return pdep64(v, m); }
uint64 pext_v64(uint64 v, uint64 m) { return pext64(v, m); }
