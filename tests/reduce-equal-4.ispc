export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    int a = aFOO[programIndex/max(1, (programCount/2))];
    RET[programIndex] = 0;
    if (programIndex >= programCount/2 && a < 4)
        RET[programIndex] = reduce_equal(a) ? 1 : 0; 
    else
        RET[programIndex] = 1;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
