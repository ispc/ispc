// rule: skip on arch=genx32
// rule: skip on arch=genx64

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform buf = uniform new uniform float[32l*32l];
    for (uniform int i = 0; i < 32l*32l; ++i)
        buf[i] = i;

    #pragma ignore warning(perf)
    float a = buf[4*programIndex];
    #pragma ignore warning(perf)
    buf[4*programIndex+1] = 0;
    #pragma ignore warning(perf)
    buf[4*programIndex+3] = 0;
    #pragma ignore warning(perf)
    float b = buf[4*programIndex+1];
    #pragma ignore warning(perf)
    float c = buf[4*programIndex+2];
    #pragma ignore warning(perf)
    float d = buf[4*programIndex+3];

    RET[programIndex] = a+b+c+d;;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 4 * programIndex + 4 * programIndex + 2;
}
