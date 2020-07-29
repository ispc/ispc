export void f_f(uniform float RET[], uniform float aFOO[]) {
    float x = aFOO[programIndex];
    float d, invsqrt = rsqrt_fast(x);
    d = (x * (invsqrt * invsqrt)) - 1.;
    if (d < 0.) d = -d;

    // The worst precision is demonstrated by ARM, it requires 1e-2 to pass.
    RET[programIndex] = (d > 1e-2) ? 1 : 0;
}


export void result(uniform float RET[]) {
       RET[programIndex] = 0;
}
