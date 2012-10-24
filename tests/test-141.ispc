
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = (exp(-log(1/a)) - a) < 1e-7 ? 1 : 0;
}

export void result(uniform float RET[4]) {
    RET[programIndex] = 1;
}
