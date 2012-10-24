
export uniform int width() { return programCount; }



export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    uniform int x = (1 << 4) - ~0xf0f0f0f0 + (2 * 8 / 2);
    static uniform int y = (1 << 4) - ~0xf0f0f0f0 + (2 * 8 / 2);
    RET[programIndex] = (x == y) ? 1. : 0.;
}

export void result(uniform float RET[]) { RET[programIndex] = 1.; }
