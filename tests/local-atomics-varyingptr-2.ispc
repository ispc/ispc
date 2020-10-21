uniform unsigned int32 s[programCount];

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = 0;
    float delta = 1;
    atomic_add_local(&s[programCount-1-programIndex], programIndex);
    #pragma ignore warning(perf)
    RET[programIndex] = s[programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = programCount-1-programIndex;
}
