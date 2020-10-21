uniform unsigned int32 s[programCount];

export void f_f(uniform float RET[], uniform float aFOO[]) {
    for (uniform int i = 0; i < programCount; ++i)
        s[i] = 1234;
    float a = aFOO[programIndex];
    float b = 0;
    float delta = 1;
    #pragma ignore warning(perf)
    a = atomic_max_local(&s[programIndex], programIndex);
    RET[programIndex] = a;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1234;
}
