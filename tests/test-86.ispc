float f() { return 2.; }
export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = 1. + f() + 2.; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 5.000000;
}
