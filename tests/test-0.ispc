export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = a+a; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2 * programIndex;
}
