uniform int32 s = 0;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = atomic_or_global(&s, (1ul<<min(programIndex,30)));
    RET[programIndex] = s;
}

export void result(uniform float RET[]) {
    RET[programIndex] = (1ul<<min(programCount,31))-1;
}
