// Check that atomic operations of llvm.ispc.atomicrmw on short vectors that
// fit in a native atomic width compile to object files on AArch64, also with
// LSE atomics, which have no vector forms (https://github.com/ispc/ispc/issues/4004).

// RUN: %{ispc} --arch=aarch64 --target=neon-i32x8 --nowrap --nostdlib -O2 --emit-obj %s -o %t.o
// RUN: %{ispc} --arch=aarch64 --target=neon-i32x8 --cpu=cortex-a78 --nowrap --nostdlib -O0 --emit-obj %s -o %t.o
// RUN: %{ispc} --arch=aarch64 --target=neon-i32x8 --cpu=cortex-a78 --nowrap --nostdlib -O2 --emit-obj %s -o %t.o

// REQUIRES: ARM_ENABLED && LLVM_23_0+

uniform int<2> atomic_add(uniform int8 *uniform p, uniform int<2> a) { return @llvm.ispc.atomicrmw.add.seq_cst(p, a); }

uniform uint16<4> atomic_umax(uniform int8 *uniform p, uniform uint16<4> a) {
    return @llvm.ispc.atomicrmw.umax.acquire(p, a);
}

uniform int8<2> atomic_xor(uniform int8 *uniform p, uniform int8<2> a) { return @llvm.ispc.atomicrmw.xor.monotonic(p, a); }

uniform float<2> atomic_fadd(uniform int8 *uniform p, uniform float<2> a) {
    return @llvm.ispc.atomicrmw.fadd.seq_cst(p, a);
}
