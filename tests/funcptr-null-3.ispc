typedef float (*FuncType)(float, float);

float foo(float a, float b) {
    return a+b;
}

static float bar(float a, float b) {
    return a < b ? a : b;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = aFOO[0]-1;
    FuncType func = &foo;
    if (a == 2)
        func = NULL;
    if (func != NULL)
        RET[programIndex] = func(a, b);
    else
        RET[programIndex] = -1;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 + 1*programIndex;
    RET[1] = -1;
}
