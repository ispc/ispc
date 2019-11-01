export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    bool b = (a == 1.);
    RET[programIndex] = (int)b;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = 1;
}
