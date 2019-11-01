export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int8 x[2*programCount];
    for (uniform int i = 0; i < 2*programCount; ++i)
        x[i] = 0xff;
    #pragma ignore warning(perf)
    unsigned int8 val = aFOO[programIndex];
    x[2+programIndex] = val;
    RET[programIndex] = x[1+programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
    RET[0] = -1.;
}
