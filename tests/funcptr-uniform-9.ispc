typedef float (*FuncType)(float, float);

float foo(float a, float b) {
    return a+b;
}

static float bar(float a, float b) {
    if (a < b)
        return a;
    else
        return b;
}


export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform FuncType func[] = { foo, bar };
    float a = aFOO[programIndex]; 
    float b = aFOO[0]-1; // == 0
    RET[programIndex] = ((a == 1) ? bar : foo)(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex+1;
    RET[0] = 0;
}
