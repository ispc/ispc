
export uniform int width() { return programCount; }

typedef float (*FuncType)(float, float);

float foo(float a, float b) {
    return a+b;
}

static float bar(float a, float b) {
    return min(a, b);
}


export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform FuncType func[] = { bar, foo };
    float a = aFOO[programIndex]; 
    float b = aFOO[0]-1; // == 0
    func[min(a, 0)] = foo;
    RET[programIndex] = func[0](a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex+1;
}
