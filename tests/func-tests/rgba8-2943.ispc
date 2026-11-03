#include "test_static.isph"
// rule: skip on arch=xe64
// The RGBA8 difference kernel from discussion #2943: 4 contiguous bytes per
// pixel, compared with another image, and a masked store of a marker color
// for the pixels that differ. The source image ends at the end of aFOO and
// the destination image starts at the start of vint, so a read past the end
// of the source or before the start of the destination faults (the harness
// puts guard pages there).

void difference_rgba32(const uniform uint8 source[], const uniform uint8 destination[], uniform uint8 output[],
                       uniform uint32 count) {
    foreach (index = 0 ... count) {
        if (source[index * 4 + 0] != destination[index * 4 + 0] ||
            source[index * 4 + 1] != destination[index * 4 + 1] ||
            source[index * 4 + 2] != destination[index * 4 + 2] ||
            source[index * 4 + 3] != destination[index * 4 + 3]) {
            output[index * 4 + 0] = 255;
            output[index * 4 + 1] = 0;
            output[index * 4 + 2] = 255;
            output[index * 4 + 3] = 255;
        }
    }
}

// Pixels whose channel c differs, for c = k % 5 (k % 5 == 4: none).
static inline uniform bool differs(uniform int k, uniform int c) { return k % 5 == c; }

task void f_fi(uniform float RET[], uniform float aFOO[], uniform int vint[]) {
    // 253 pixels, so the loop has a partial last iteration; the source ends
    // at the last byte of aFOO.
    uniform int n = 253;
    uniform uint8 *uniform src = (uniform uint8 * uniform) aFOO + 1024 - 4 * n;
    uniform uint8 *uniform dst = (uniform uint8 * uniform) vint;
    uniform uint8 out[1024];
    for (uniform int k = 0; k < n; ++k) {
        for (uniform int c = 0; c < 4; ++c) {
            src[4 * k + c] = (uint8)(k * 7 + c * 31);
            dst[4 * k + c] = (uint8)(k * 7 + c * 31 + (differs(k, c) ? 1 : 0));
        }
    }
    for (uniform int b = 0; b < 1024; ++b)
        out[b] = 17;
    difference_rgba32(src, dst, out, n);

    uniform int err = 0;
    for (uniform int k = 0; k < 256; ++k) {
        uniform bool marked = k < n && k % 5 != 4;
        for (uniform int c = 0; c < 4; ++c) {
            uniform uint8 want = marked ? (c == 1 ? 0 : 255) : 17;
            err += out[4 * k + c] != want ? 1 : 0;
        }
    }
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
