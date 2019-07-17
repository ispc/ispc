export void f_f(uniform float RET[], uniform float aFOO[]) {
    float x = aFOO[programIndex];
    float d, invsqrt = rsqrt_fast(x);
    d = (x * (invsqrt * invsqrt)) - 1.;
    if (d < 0.) d = -d;
    if (d < 1e-3)
    {
        RET[programIndex] = 1;
    }
    else
    {
        RET[programIndex] = 0;
    }
}


export void result(uniform float RET[]) {
       RET[programIndex] = 1;
}
