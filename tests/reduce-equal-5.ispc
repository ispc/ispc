export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    int a = aFOO[programIndex&1];
    RET[programIndex] = 1;
    if (programIndex & 1)
        RET[programIndex] = reduce_equal(a) ? 1 : 0; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
