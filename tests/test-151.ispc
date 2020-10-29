// rule: skip on cpu=TGLLP

export void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = aFOO[programIndex]; 
    RET[programIndex] = floor(a+.99999); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 +(double) programIndex;
}
