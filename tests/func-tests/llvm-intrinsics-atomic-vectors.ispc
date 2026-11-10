#include "test_static.isph"

// Atomic operations of llvm.ispc.atomicrmw on short vectors that fit in a
// native atomic width of the target, which LowerISPCIntrinsicsPass expands to
// loops of cmpxchg (https://github.com/ispc/ispc/issues/4004). The vectors have
// two elements, so that their types differ from varying types on all targets.
// Integer operations on vectors require LLVM 23.

static uniform bool check_fadd() {
    uniform int64 storage = 0;
    uniform float *uniform mem = (uniform float *uniform)&storage;
    mem[0] = 1.5;
    mem[1] = 2;
    uniform float<2> a = {0.25, -4};
    uniform float<2> old = @llvm.ispc.atomicrmw.fadd.seq_cst((uniform int8 *uniform)mem, a);
    return old[0] == 1.5 && old[1] == 2 && mem[0] == 1.75 && mem[1] == -2;
}

#if LLVM_VERSION_MAJOR >= 23
static uniform bool check_add() {
    uniform int64 storage = 0;
    uniform int32 *uniform mem = (uniform int32 *uniform)&storage;
    mem[0] = 10;
    mem[1] = -20;
    uniform int<2> a = {3, 5};
    uniform int<2> old = @llvm.ispc.atomicrmw.add.seq_cst((uniform int8 *uniform)mem, a);
    return old[0] == 10 && old[1] == -20 && mem[0] == 13 && mem[1] == -15;
}

static uniform bool check_max() {
    uniform int32 storage = 0;
    uniform int16 *uniform mem = (uniform int16 *uniform)&storage;
    mem[0] = 1;
    mem[1] = -1;
    uniform int16<2> a = {5, -2};
    uniform int16<2> old = @llvm.ispc.atomicrmw.max.acq_rel((uniform int8 *uniform)mem, a);
    return old[0] == 1 && old[1] == -1 && mem[0] == 5 && mem[1] == -1;
}

static uniform bool check_xor() {
    uniform int16 storage = 0;
    uniform int8 *uniform mem = (uniform int8 *uniform)&storage;
    mem[0] = 0x0F;
    mem[1] = 0x33;
    uniform int8<2> a = {-1, 0x0F};
    uniform int8<2> old = @llvm.ispc.atomicrmw.xor.monotonic(mem, a);
    return old[0] == 0x0F && old[1] == 0x33 && mem[0] == -16 && mem[1] == 0x3C;
}
#else
static uniform bool check_add() { return true; }
static uniform bool check_max() { return true; }
static uniform bool check_xor() { return true; }
#endif

task void f_v(uniform float RET[]) {
    RET[programIndex] = (check_fadd() ? 1 : 0) + (check_add() ? 2 : 0) + (check_max() ? 4 : 0) + (check_xor() ? 8 : 0);
}

task void result(uniform float RET[]) { RET[programIndex] = 15; }
