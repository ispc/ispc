uniform int32 s = 0xbeef;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = atomic_swap_global(&s, programIndex);
    RET[programIndex] = reduce_max(b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0xbeef;
}
