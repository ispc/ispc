
export uniform int width() { return programCount; }



export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    uniform int x = (170 >> 4) % 5;
    static uniform int y = (170 >> 4) % 5;
    RET[programIndex] = (x == y) ? 1. : 0.;
}

export void result(uniform float RET[]) { RET[programIndex] = 1.; }
