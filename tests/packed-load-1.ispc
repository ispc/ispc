export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform unsigned int a[programCount];
    #pragma ignore warning(perf)
    a[programIndex] = aFOO[programIndex];
    unsigned int aa;
    packed_load_active(a, &aa);
    #pragma ignore warning(perf)
    RET[programIndex] = aa;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
