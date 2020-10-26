uniform int32 s = 0;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 a = aFOO[programIndex];
    int32 b = 0;
    if (programIndex & 1)
        b = atomic_max_global(&s, a);
    RET[programIndex] = s;
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programCount == 1) ? 0 : programCount;
}
