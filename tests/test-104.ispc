
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    float x = aFOO[programIndex&0x3]; 
    float d, ix; 
    ix = rcp(x);
    d = (ix - 1. / x);
    d = (d < 0) ? -d : d;
    RET[programIndex] = (d < 1e-7) ? 1. : 0.; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1.;
}
