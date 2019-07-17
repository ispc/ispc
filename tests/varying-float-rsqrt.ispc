export void f_f(uniform float RET[], uniform float aFOO[]) {
    varying float x = aFOO[programIndex];
    varying float d, invsqrt = rsqrt(x);
    d = (x * (invsqrt * invsqrt)) - 1.;
    if (d < 0.) d = -d;
    if (d < 1e-5)
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
