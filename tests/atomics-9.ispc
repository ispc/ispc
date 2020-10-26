uniform unsigned int32 s = 0;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = 0;
    int32 delta = 1;
    if (programIndex < 2) {
        #pragma ignore warning(perf)
        b = atomic_add_global(&s, delta);
    }
    RET[programIndex] = reduce_add(b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programCount == 1) ? 0 : 1;
}
