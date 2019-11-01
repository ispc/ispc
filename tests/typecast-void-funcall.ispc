uniform float add(float a, float b, uniform float * uniform result) {
    result[programIndex] = a+b;
    return 1;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = 0.; b = a; 
    (void)add(a, b, RET);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
