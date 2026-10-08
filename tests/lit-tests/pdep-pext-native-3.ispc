// pdep32/pext32/pdep64/pext64 with dynamic operands use the native BMI2
// instructions on AVX10 targets, which inherit from the AVX-512 roots.
// RUN: %{ispc} %s --target=avx10.2nvl-x4 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx10.2nvl-x8 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx10.2nvl-x16 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx10.2nvl-x32 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx10.2nvl-x64 --nowrap --emit-asm -o - | FileCheck %s

// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_22_0+

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
