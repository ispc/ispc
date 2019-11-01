// rule: skip on arch=genx32
// rule: skip on arch=genx64

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform buf = uniform new uniform float[32l*32l];
    for (uniform int i = 0; i < 32l*32l; ++i)
        buf[i] = i;

    #pragma ignore warning(perf)
    RET[programIndex] = buf[(programIndex >> 2) * 16 + (programIndex & 3)];
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programIndex >> 2) * 16 + (programIndex & 3);
}
