export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = round(a+.5); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programIndex / 2) * 2 + 2;
}
