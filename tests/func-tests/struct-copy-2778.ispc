#include "test_static.isph"
// rule: skip on arch=xe64
// Copies of an 8-byte struct of byte fields (#2778): contiguous, indexed and
// masked. The test checks every byte of the copies and that the elements
// that are not copied keep their value. The source structs end at the end
// of aFOO, so a read past them faults (the harness puts a guard page there).

struct Foo {
    uint8 a, b, c, d, e, f, g, h;
};

#define N 128       // structs in aFOO
#define NBYTES 1024 // 8 * N (8 * N folds in 8 bits on targets with 8-bit data)

static inline uniform uint8 src_byte(uniform int k, uniform int b) { return (uint8)(k * 37 + b * 101 + 3); }

void copy(uniform Foo dst[], uniform Foo src[], uniform int first) {
    dst[first + programIndex] = src[first + programIndex];
}

void copy_idx(uniform Foo dst[], uniform Foo src[], int i) { dst[i] = src[i]; }

// The number of wrong bytes in dst: the struct k should be a copy of src[k]
// if copied[k], else all bytes 0xee.
uniform int wrong(uniform Foo dst[], uniform bool copied[]) {
    uniform int err = 0;
    for (uniform int k = 0; k < N; ++k) {
        uniform uint8 *uniform d = (uniform uint8 * uniform) & dst[k];
        for (uniform int b = 0; b < 8; ++b)
            err += d[b] != (copied[k] ? src_byte(k, b) : 0xee) ? 1 : 0;
    }
    return err;
}

void reset(uniform Foo dst[], uniform bool copied[]) {
    uniform uint8 *uniform d = (uniform uint8 * uniform) & dst[0];
    for (uniform int b = 0; b < NBYTES; ++b)
        d[b] = 0xee;
    for (uniform int k = 0; k < N; ++k)
        copied[k] = false;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int r = (int)aFOO[programIndex] - 1;
    uniform Foo *uniform src = (uniform Foo * uniform) aFOO;
    uniform uint8 *uniform s = (uniform uint8 * uniform) aFOO;
    for (uniform int k = 0; k < N; ++k)
        for (uniform int b = 0; b < 8; ++b)
            s[8 * k + b] = src_byte(k, b);
    uniform Foo dst[N];
    uniform bool copied[N];
    uniform int err = 0;

    // Contiguous: the last programCount structs.
    reset(dst, copied);
    uniform int first = N - programCount;
    #pragma ignore warning(perf)
    copy(dst, src, first);
    for (uniform int k = first; k < N; ++k)
        copied[k] = true;
    err += wrong(dst, copied);

    // Indexed: struct N - 1 - 3r (distinct for every lane).
    reset(dst, copied);
    int i = N - 1 - 3 * r;
    if (i >= 0) {
        #pragma ignore warning(perf)
        copy_idx(dst, src, i);
    }
    foreach_active (l) {
        uniform int k = extract(i, l);
        if (k >= 0)
            copied[k] = true;
    }
    err += wrong(dst, copied);

    // Masked: every third lane of the last programCount structs.
    reset(dst, copied);
    if (r % 3 == 0) {
        #pragma ignore warning(perf)
        copy_idx(dst, src, first + r);
        foreach_active (l)
            copied[first + extract(r, l)] = true;
    }
    err += wrong(dst, copied);

    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
