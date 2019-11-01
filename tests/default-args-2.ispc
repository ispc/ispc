float foo(uniform float x, float f = 1) { return f+x; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = foo(b);
}

export void result(uniform float RET[]) { RET[programIndex] = 6; }
