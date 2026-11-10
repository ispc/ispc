// Check that atomic operations of llvm.ispc.atomicrmw accept short vectors of
// integers, which the LLVM verifier supports since LLVM 23, and that the
// operations on vectors are expanded to loops of cmpxchg of integers of the
// same size, which the backends support also for vectors that fit in a native
// atomic width (https://github.com/ispc/ispc/issues/4004).

// RUN: %{ispc} --target=avx2-i32x8 --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx2-i8x32 --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx2-i32x8 --nowrap --nostdlib -O0 --emit-obj %s -o %t.o
// RUN: %{ispc} --target=avx2-i32x8 --nowrap --nostdlib -O2 --emit-obj %s -o %t.o
// RUN: %{ispc} --target=avx512skx-x16 --nowrap --nostdlib -O2 --emit-obj %s -o %t.o
// The expanded instructions have the debug location of the call.
// RUN: %{ispc} --target=avx2-i32x8 --nowrap --nostdlib -O0 -g --emit-llvm-text %s -o - | FileCheck %s --check-prefix=CHECK-DBG
// RUN: %{ispc} --target=avx2-i32x8 --nowrap --nostdlib -O0 -g --emit-obj %s -o %t.o

// REQUIRES: X86_ENABLED && LLVM_23_0+

// CHECK-LABEL: @atomic_add___
// CHECK: [[INIT_INT:%.*]] = load atomic i128, ptr %p monotonic, align 16
// CHECK: [[INIT:%.*]] = bitcast i128 [[INIT_INT]] to <4 x i32>
// CHECK: [[LOADED:%.*]] = phi <4 x i32> [ [[INIT]], %{{.*}} ], [ [[NEW_LOADED:%.*]], %atomicrmw.start ]
// CHECK: [[NEW:%.*]] = add <4 x i32> [[LOADED]], %{{.*}}
// CHECK: [[CMP:%.*]] = bitcast <4 x i32> [[LOADED]] to i128
// CHECK: [[NEWINT:%.*]] = bitcast <4 x i32> [[NEW]] to i128
// CHECK: cmpxchg ptr %p, i128 [[CMP]], i128 [[NEWINT]] seq_cst seq_cst, align 16
// CHECK: [[NEW_LOADED]] = bitcast i128 %{{.*}} to <4 x i32>
// CHECK: br i1 %{{.*}}, label %atomicrmw.end, label %atomicrmw.start
// CHECK-NOT: = atomicrmw
// CHECK-DBG-LABEL: @atomic_add___
// CHECK-DBG: load atomic i128, ptr %{{.*}} monotonic, align 16, !dbg [[DBG:![0-9]+]]
// CHECK-DBG: cmpxchg ptr %{{.*}}, i128 %{{.*}}, i128 %{{.*}} seq_cst seq_cst, align 16, !dbg [[DBG]]
// CHECK-DBG: br i1 %{{.*}}, label %atomicrmw.end, label %atomicrmw.start, !dbg [[DBG]]
uniform int<4> atomic_add(uniform int8 *uniform p, uniform int<4> a) { return @llvm.ispc.atomicrmw.add.seq_cst(p, a); }

// CHECK-LABEL: @atomic_umax___
// CHECK: call <2 x i64> @llvm.umax.v2i64(
// CHECK: cmpxchg ptr %p, i128 %{{.*}}, i128 %{{.*}} seq_cst seq_cst, align 16
// CHECK-NOT: = atomicrmw
uniform uint64<2> atomic_umax(uniform int8 *uniform p, uniform uint64<2> a) {
    return @llvm.ispc.atomicrmw.umax.seq_cst(p, a);
}

// The vectors below fit in the native atomic width.
// CHECK-LABEL: @atomic_add_narrow___
// CHECK: load atomic i64, ptr %p monotonic, align 8
// CHECK: add <2 x i32>
// CHECK: cmpxchg ptr %p, i64 %{{.*}}, i64 %{{.*}} seq_cst seq_cst, align 8
// CHECK-NOT: = atomicrmw
uniform int<2> atomic_add_narrow(uniform int8 *uniform p, uniform int<2> a) {
    return @llvm.ispc.atomicrmw.add.seq_cst(p, a);
}

// The failure ordering is the strongest one allowed for the ordering.
// CHECK-LABEL: @atomic_nand_narrow___
// CHECK: and <2 x i16>
// CHECK: xor <2 x i16>
// CHECK: cmpxchg ptr %p, i32 %{{.*}}, i32 %{{.*}} acq_rel acquire, align 4
// CHECK-NOT: = atomicrmw
uniform int16<2> atomic_nand_narrow(uniform int8 *uniform p, uniform int16<2> a) {
    return @llvm.ispc.atomicrmw.nand.acq_rel(p, a);
}

// CHECK-LABEL: @atomic_smin_narrow___
// CHECK: call <2 x i8> @llvm.smin.v2i8(
// CHECK: cmpxchg ptr %p, i16 %{{.*}}, i16 %{{.*}} monotonic monotonic, align 2
// CHECK-NOT: = atomicrmw
uniform int8<2> atomic_smin_narrow(uniform int8 *uniform p, uniform int8<2> a) {
    return @llvm.ispc.atomicrmw.min.monotonic(p, a);
}

// CHECK-LABEL: @atomic_fmax_narrow___
// CHECK: call <2 x float> @llvm.maxnum.v2f32(
// CHECK: cmpxchg ptr %p, i64 %{{.*}}, i64 %{{.*}} release monotonic, align 8
// CHECK-NOT: = atomicrmw
uniform float<2> atomic_fmax_narrow(uniform int8 *uniform p, uniform float<2> a) {
    return @llvm.ispc.atomicrmw.fmax.release(p, a);
}

// Scalar operations are not expanded.
// CHECK-LABEL: @atomic_add_scalar___
// CHECK: atomicrmw add ptr %p, i32 %a seq_cst, align 4
// CHECK-NOT: cmpxchg
uniform int atomic_add_scalar(uniform int8 *uniform p, uniform int a) { return @llvm.ispc.atomicrmw.add.seq_cst(p, a); }
