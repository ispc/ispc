// pdep/pext compile on Xe targets, which define the software builtins in
// their separate builtin library.
// RUN: %{ispc} %s --target=xehpg-x8 --device=acm-g10 --nowrap --emit-llvm-text -o - | FileCheck %s --implicit-check-not="{{llvm.(x86.bmi|pdep|pext)}}" --implicit-check-not="declare {{.*}}@__pdep_" --implicit-check-not="declare {{.*}}@__pext_"
// RUN: %{ispc} %s --target=xe2hpg-x16 --nowrap --emit-llvm-text -o - | FileCheck %s --implicit-check-not="{{llvm.(x86.bmi|pdep|pext)}}" --implicit-check-not="declare {{.*}}@__pdep_" --implicit-check-not="declare {{.*}}@__pext_"
// RUN: %{ispc} %s --target=xehpg-x8 --device=acm-g10 --emit-spirv -o %t.spv

// REQUIRES: XE_ENABLED

// CHECK: define

uniform uint32 pdep_u32(uniform uint32 v, uniform uint32 m) { return pdep32(v, m); }
uniform uint32 pext_u32(uniform uint32 v, uniform uint32 m) { return pext32(v, m); }
uniform uint64 pdep_u64(uniform uint64 v, uniform uint64 m) { return pdep64(v, m); }
uniform uint64 pext_u64(uniform uint64 v, uniform uint64 m) { return pext64(v, m); }
uint32 pdep_v32(uint32 v, uint32 m) { return pdep32(v, m); }
uint32 pext_v32(uint32 v, uint32 m) { return pext32(v, m); }
uint64 pdep_v64(uint64 v, uint64 m) { return pdep64(v, m); }
uint64 pext_v64(uint64 v, uint64 m) { return pext64(v, m); }
