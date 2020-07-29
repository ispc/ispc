export void f_f(uniform float RET[], uniform float aFOO[]) {
    #pragma ignore warning(perf)
    varying float x = aFOO[programIndex];
    varying float d, ix;
    ix = rcp_fast(x);
    d = (ix - 1. / x);
    d = (d < 0) ? -d : d;
    RET[programIndex] = (d < 1e-3) ? 1. : 0.;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1.;
}
