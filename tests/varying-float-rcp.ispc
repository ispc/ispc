export void f_f(uniform float RET[], uniform float aFOO[]) {
    #pragma ignore warning(perf)
    varying float x = aFOO[programIndex];
    varying float d, ix;
    ix = rcp(x);
    d = (ix - 1. / x);
    d = (d < 0) ? -d : d;
    if (d < 1e-7)
    {
        RET[programIndex] = 1.;
    }
    else
    {
        RET[programIndex] = 0.;
    }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1.;
}
