typedef float (*FuncType)(float, float);

float foo(float a, float b) {
    return a+b;
}

static float bar(float a, float b) {
    return min(a, b);
}


export void f_f(uniform float RET[], uniform float aFOO[]) {
    FuncType func;
    float a = aFOO[programIndex]; 
    float b = aFOO[0]-1;
    func = (a == 2) ? foo : bar;
    RET[programIndex] = func(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[1] = 2;
}
