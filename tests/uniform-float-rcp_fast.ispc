export void f_f(uniform float RET[], uniform float aFOO[]) {
    #pragma ignore warning(perf)
    uniform float x = aFOO[programCount/2]; //9
    uniform float d, ix;
    ix = rcp_fast(x); //9
    d = (ix - 1. / x);
    d = (d < 0) ? -d : d;
    RET[programIndex] = (d < 1e-3) ? 1. : 0.;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1.;
}
