export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float x[2*programCount];
    for (uniform int i = 0; i < 2*programCount; ++i)
        x[i] = 0;
    #pragma ignore warning(perf)
    x[a] = a;
    RET[programIndex] = x[4] + x[0] + x[5];
}

    
export void result(uniform float RET[]) { 
    RET[programIndex] = 0;
    RET[3] = 4;
    RET[4] = 5;
}
