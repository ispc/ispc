export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int16 x[2*programCount+1];
    for (uniform int i = 0; i < 2*programCount+1; ++i)
        x[i] = 0xffff;
    #pragma ignore warning(perf)
    unsigned int8 val = aFOO[programIndex];
    x[2+programIndex] = val;
    RET[programIndex] = x[1+programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
    RET[0] = -1.;
}
