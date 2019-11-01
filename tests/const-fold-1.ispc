export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    #pragma ignore warning
    uniform int x = (1 << 4) - ~0xf0f0f0f0 + (2 * 8 / 2);
    #pragma ignore warning
    const static uniform int y = (1 << 4) - ~0xf0f0f0f0 + (2 * 8 / 2);
    RET[programIndex] = (x == y) ? 1. : 0.;
}

export void result(uniform float RET[]) { RET[programIndex] = 1.; }
