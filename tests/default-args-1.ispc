float foo(float f = 1) { return f; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    RET[programIndex] = foo();
}

export void result(uniform float RET[]) { RET[programIndex] = 1; }
