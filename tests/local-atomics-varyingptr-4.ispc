uniform int32 s[programCount];

export void f_f(uniform float RET[], uniform float aFOO[]) {
    for (uniform int i = 0; i < programCount; ++i)
        s[i] = -1234;
    atomic_max_local(&s[programIndex], programIndex);
    RET[programIndex] = s[programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
