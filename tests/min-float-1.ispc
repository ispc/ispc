
export uniform int width() { return programCount; }



export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    RET[programIndex] = min(10 * a, 10.f);
}

export void result(uniform float RET[]) { RET[programIndex] = 10.; }
