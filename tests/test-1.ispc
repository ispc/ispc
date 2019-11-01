export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = 0.; b = a; 
    RET[programIndex] = a+b; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
