uniform unsigned int32 s = 0;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = 0;
    if (programIndex < 30 && programIndex & 1) {
        #pragma ignore warning(perf)
        b = atomic_or_global(&s, (1ul << programIndex));
    }
    RET[programIndex] = s;
}

export void result(uniform float RET[]) {
    uniform int sum = 0;
    for (uniform int i = 0; i < min(30, programCount); ++i)
        if (i & 1)
            sum += (1ul << i);
    RET[programIndex] = sum;
}
