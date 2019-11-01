export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = a < 65.; 
}


export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
