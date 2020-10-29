// rule: skip on cpu=TGLLP

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    double a = aFOO[programIndex/2];
    RET[programIndex] = 1;
    uniform bool eq = false;
    if (a < 4)
        eq = reduce_equal(a); 
    RET[programIndex] = eq;
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programCount == 1) ? 1 : 0;
}
