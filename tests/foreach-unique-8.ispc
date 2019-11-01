export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int k = 0;
    #pragma ignore warning(perf)
    int index = aFOO[programIndex/2];
    foreach_unique(j in index) {
        if (j & 1) continue;
        ++k;
    }
    RET[programIndex] = k;
}

export void result(uniform float RET[]) {
    RET[programIndex] = programCount / 4;
}
