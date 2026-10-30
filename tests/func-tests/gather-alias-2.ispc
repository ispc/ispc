#include "test_static.isph"
// rule: skip on arch=xe64
// A scatter followed by a gather of overlapping addresses: each lane reads
// the element another lane has just written, plus elements nobody wrote.
// The gather must not be moved before the scatter or merged with an earlier
// load of the same elements.

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int r = (int)aFOO[programIndex] - 1;
    int err = 0;
    // A load of elements that the scatter then overwrites.
    uniform int base = 64;
    #pragma ignore warning(perf)
    float before = aFOO[base + 2 * r];
    // Lane r writes element base + 2 * r.
    #pragma ignore warning(perf)
    aFOO[base + 2 * r] = -(r + 1);
    // Lane r reads the element of lane r ^ 1, and the unwritten element after it.
    int o = (r ^ 1) < programCount ? (r ^ 1) : r;
    #pragma ignore warning(perf)
    float got = aFOO[base + 2 * o];
    #pragma ignore warning(perf)
    float next = aFOO[base + 2 * o + 1];
    err += before != base + 2 * r + 1 ? 1 : 0;
    err += got != -(o + 1) ? 1 : 0;
    err += next != base + 2 * o + 2 ? 1 : 0;
    // The same under a mask: only even lanes write, every lane reads.
    uniform int base2 = 192;
    if (r % 2 == 0) {
        #pragma ignore warning(perf)
        aFOO[base2 + r] = -(r + 1);
    }
    #pragma ignore warning(perf)
    float got2 = aFOO[base2 + o];
    err += got2 != (o % 2 == 0 ? -(o + 1) : base2 + o + 1) ? 1 : 0;
    RET[programIndex] = err;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
