void foo(uniform float * uniform * ret) {
    #pragma ignore warning(perf)
    uniform float *px = *ret;
    ++px;
    #pragma ignore warning
    *ret = px;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform ptr = &aFOO[0];
    foo(&ptr);
    RET[programIndex] = *ptr;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2;
}
