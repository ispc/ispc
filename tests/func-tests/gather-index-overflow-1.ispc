#include "test_static.isph"
// rule: skip on arch=xe64
// Gathers and scatters whose unsigned 32-bit byte offset is >= 2^31. The
// offset fits in 32 bits, so it is exact in both addressing modes, and it is
// zero-extended: a gather or scatter that sign-extends it (a dword one)
// computes base - 2^32 + offset instead. The base is moved back by integer
// arithmetic, so that every address lands in aFOO. The large constants are
// multiplied by `one`, which is 1 at run time, so that the compiler cannot
// fold the bias and the index back into a small offset.

// One lane per element of aFOO, in reverse order.
static inline int lane_elem(int r) { return programCount - 1 - r; }

// int8 elements: index 2^31 + k, byte offset 2^31 + k. No x86 target has a
// byte gather, so this is the emulated path. Uses bytes [512, 768) of aFOO;
// check_f uses the first 64 floats.
int check_i8(uniform float aFOO[], uniform int one, int r) {
    uniform int8 *uniform bytes = (uniform int8 * uniform) aFOO;
    uniform int8 *uniform base = (uniform int8 * uniform)((uniform int64)bytes - 0x80000000ll * one);
    unsigned int32 k = 512 + lane_elem(r) * 4;
    unsigned int32 idx = 0x80000000u * one + k;
    int err = 0;
    #pragma ignore warning(perf)
    int8 g = base[idx];
    #pragma ignore warning(perf)
    err += g != bytes[k] ? 1 : 0;
    #pragma ignore warning(perf)
    base[idx + 1] = (int8)(r + 1);
    #pragma ignore warning(perf)
    err += bytes[k + 1] != (int8)(r + 1) ? 1 : 0;
    return err;
}

// float elements: index in [2^29, 2^30), byte offset 4 * index in
// [2^31, 2^32), a native gather and scatter.
int check_f(uniform float aFOO[], uniform int one, int r) {
    uniform float *uniform base = (uniform float * uniform)((uniform int64)aFOO - 0x80000000ll * one);
    unsigned int32 k = lane_elem(r);
    unsigned int32 idx = 0x20000000u * one + k;
    int err = 0;
    #pragma ignore warning(perf)
    float g = base[idx];
    err += g != (float)(k + 1) ? 1 : 0;
    #pragma ignore warning(perf)
    base[idx] = -(float)(r + 1);
    #pragma ignore warning(perf)
    err += aFOO[k] != -(float)(r + 1) ? 1 : 0;
    return err;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int r = (int)aFOO[programIndex] - 1;
    uniform int one = (uniform int)aFOO[0];
    int err = 0;
    err += check_i8(aFOO, one, r);
    if (r % 3 != 1)
        err += check_i8(aFOO, one, r);
    err += check_f(aFOO, one, r);
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
