// Check that atomic integer operations of llvm.ispc.atomicrmw accept short
// vectors, which the LLVM verifier supports since LLVM 23
// (https://github.com/ispc/ispc/issues/4004).

// RUN: %{ispc} --target=avx2-i32x8 --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx2-i8x32 --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s

// REQUIRES: X86_ENABLED && LLVM_23_0+

// CHECK-LABEL: @atomic_add___
// CHECK: atomicrmw add ptr %p, <4 x i32> %{{.*}} seq_cst, align 16
uniform int<4> atomic_add(uniform int8 *uniform p, uniform int<4> a) { return @llvm.ispc.atomicrmw.add.seq_cst(p, a); }

// CHECK-LABEL: @atomic_umax___
// CHECK: atomicrmw umax ptr %p, <2 x i64> %{{.*}} seq_cst, align 16
uniform uint64<2> atomic_umax(uniform int8 *uniform p, uniform uint64<2> a) {
    return @llvm.ispc.atomicrmw.umax.seq_cst(p, a);
}
