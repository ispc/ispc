export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    RET[programIndex] = a < 3.;
}

export void result(uniform float RET[]) { RET[programIndex] = 0; RET[0] = RET[1] = 1; }
