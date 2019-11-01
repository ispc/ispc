export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int16 x[2*programCount];
    for (uniform int i = 0; i < 2*programCount; ++i)
        x[i] = 0xffff;
    #pragma ignore warning(perf)
    unsigned int16 val = aFOO[programIndex];
    if (programIndex & 1)
        x[2+programIndex] = val;
    #pragma ignore warning(perf)
    RET[programIndex] = x[1+programIndex];
}

export void result(uniform float RET[]) {
    if (programIndex & 1)
        RET[programIndex] = 65535;
    else
        RET[programIndex] = programIndex;
    RET[0] = 65535;
}
