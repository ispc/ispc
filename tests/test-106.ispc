export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    RET[programIndex] = clamp(3*a, 5., 10.);
}


export void result(uniform float RET[]) {
    RET[programIndex] = 10;
    RET[0] = 5;
    RET[1] = 6;
    RET[2] = 9;
}
