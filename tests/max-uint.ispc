export void f_f(uniform float result[], uniform float aa[]) {
    #pragma ignore warning(perf)
    unsigned int i = (unsigned int)aa[programIndex];
    #pragma ignore warning(perf)
    result[programIndex] = max((unsigned int)100, i);
}

export void result(uniform float r[]) { r[programIndex] = 100; }

