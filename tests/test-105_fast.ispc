export void f_f(uniform float RET[], uniform float aFOO[]) {
    float x = aFOO[programIndex];
    float d, invsqrt = rsqrt_fast(x);
    d = (x * (invsqrt * invsqrt)) - 1.;
    if (d < 0.) d = -d;
    RET[programIndex] = d > 1e-2;
}


export void result(uniform float RET[]) {
       RET[programIndex] = 0;
}
