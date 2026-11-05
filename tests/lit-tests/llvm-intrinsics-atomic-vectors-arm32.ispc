// Check that atomic operations of llvm.ispc.atomicrmw on short vectors that
// fit in a native atomic width compile to object files on 32-bit ARM
// (https://github.com/ispc/ispc/issues/4004).

// RUN: %{ispc} --arch=arm --target=neon-i32x4 --nowrap --nostdlib -O0 --emit-obj %s -o %t.o
// RUN: %{ispc} --arch=arm --target=neon-i32x4 --nowrap --nostdlib -O2 --emit-obj %s -o %t.o

// REQUIRES: ARM32_ENABLED && LINUX_HOST && LLVM_23_0+

uniform int<2> atomic_add(uniform int8 *uniform p, uniform int<2> a) { return @llvm.ispc.atomicrmw.add.seq_cst(p, a); }

uniform int16<2> atomic_umin(uniform int8 *uniform p, uniform int16<2> a) {
    return @llvm.ispc.atomicrmw.umin.acquire(p, a);
}

uniform int8<2> atomic_or(uniform int8 *uniform p, uniform int8<2> a) { return @llvm.ispc.atomicrmw.or.monotonic(p, a); }

uniform float<2> atomic_fadd(uniform int8 *uniform p, uniform float<2> a) {
    return @llvm.ispc.atomicrmw.fadd.seq_cst(p, a);
}
