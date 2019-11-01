export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    int64 a = aFOO[programIndex/2];
    RET[programIndex] = 1;
    if (a == 1)
        RET[programIndex] = reduce_equal(a) ? 1 : 0; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
