int64 zero = 0;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int64 a[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        a[i] = aFOO[i];
    
    int g = 0;
    if (programIndex < 2) {
        #pragma ignore warning(perf)
        g = a[programIndex+zero];
    }
    RET[programIndex] = g; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = 1;
    RET[1] = 2;
}
