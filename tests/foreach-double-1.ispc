#include "../test_static.isph"
// rule: skip on arch=xe64

uniform double one = 1;

void copy(uniform double dst[], uniform double src[], uniform int count) {
    foreach (i = 0 ... count)
        dst[i] = one * src[i];
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int count = 200 + aFOO[1];
    uniform double * uniform src = uniform new uniform double[count];
    for (uniform int i = 0; i < count; ++i)
        src[i] = i;

    uniform double * uniform dst = uniform new uniform double[count];
    copy(dst, src, count);

    uniform int errors = 0;
    for (uniform int i = 0; i < count; ++i)
        if (dst[i] != src[i])
            ++errors;

    RET[programIndex] = errors; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
