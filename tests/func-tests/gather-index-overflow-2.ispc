#include "test_static.isph"
// rule: skip on addressing=*
// rule: run on addressing=64
// rule: skip on arch=xe64
// Gathers and scatters of floats whose byte offset is 2^31 or more. Only
// 64-bit addressing computes such offsets exactly; with 32-bit addressing
// they wrap. The base is moved back by integer arithmetic, so that every
// address lands in aFOO. The bias and the large index constants (except the
// displacement in check_disp, which must stay a constant) are multiplied by
// `one`, which is 1 at run time, so that the compiler cannot fold them back
// into a small offset.

// One lane per element of aFOO, in reverse order.
static inline int lane_elem(int r) { return programCount - 1 - r; }

// Unsigned 32-bit index 2^31 + k: byte offset 2^33 + 4k.
int check_u32(uniform float aFOO[], uniform int one, int r) {
    uniform float *uniform base = (uniform float * uniform)((uniform int64)aFOO - 0x200000000ll * one);
    unsigned int32 k = lane_elem(r);
    unsigned int32 idx = 0x80000000u * one + k;
    int err = 0;
    #pragma ignore warning(perf)
    err += base[idx] != aFOO[k] ? 1 : 0;
    #pragma ignore warning(perf)
    base[idx] = aFOO[k] + 1000;
    return err;
}

// Signed 32-bit index 2^29 + k: the index fits in 31 bits, the byte offset
// 2^31 + 4k does not.
int check_s32(uniform float aFOO[], uniform int one, int r) {
    uniform float *uniform base = (uniform float * uniform)((uniform int64)aFOO - 0x80000000ll * one);
    int32 k = lane_elem(r);
    int32 idx = 0x20000000 * one + k;
    int err = 0;
    #pragma ignore warning(perf)
    err += base[idx] != aFOO[k] ? 1 : 0;
    #pragma ignore warning(perf)
    base[idx] = aFOO[k] + 1000;
    return err;
}

// A constant displacement: p[k + 0x20000000] with a small varying k. The
// compiler factors the constant out of the index; its byte offset is 2^31.
int check_disp(uniform float aFOO[], uniform int one, int r) {
    uniform float *uniform base = (uniform float * uniform)((uniform int64)aFOO - 0x80000000ll * one);
    int32 k = lane_elem(r);
    int err = 0;
    #pragma ignore warning(perf)
    err += base[k + 0x20000000] != aFOO[k] ? 1 : 0;
    #pragma ignore warning(perf)
    base[k + 0x20000000] = aFOO[k] + 1000;
    return err;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int r = (int)aFOO[programIndex] - 1;
    uniform int one = (uniform int)aFOO[0];
    int k = lane_elem(r);
    float orig = k + 1;
    int err = 0;
    // Each scatter adds 1000 to the lane's element; the next gather checks
    // that it reached the right place.
    err += check_u32(aFOO, one, r);
    err += check_s32(aFOO, one, r);
    err += aFOO[k] != orig + 2000 ? 1 : 0;
    if (r % 3 != 1)
        err += check_disp(aFOO, one, r);
    err += aFOO[k] != (r % 3 != 1 ? orig + 3000 : orig + 2000) ? 1 : 0;
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
