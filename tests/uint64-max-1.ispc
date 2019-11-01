export void f_f(uniform float RET[], uniform float aFOO[]) {
    #pragma ignore warning(perf)
    unsigned int64 a = aFOO[programIndex];
    unsigned int64 b = 2;
    #pragma ignore warning(perf)
    RET[programIndex] = max(a,b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = max(2, 1 + programIndex);
}
