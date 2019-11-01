export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = ceil(a-1e-5); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
