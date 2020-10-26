uniform int32 s = 1234;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = 0;
    if (programIndex & 1) {
        b = atomic_swap_global(&s, programIndex);
    }
    RET[programIndex] = reduce_add(b) + s;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1234 + reduce_add(programIndex & 1 ? programIndex : 0);
}
