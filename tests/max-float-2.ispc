export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = max(-10 * (a-3), .1f);
    RET[programCount-1] = max(-10 * b, 2.f);
}

export void result(uniform float RET[]) {
    RET[programIndex] = .1;
    RET[0] = 20;
    RET[1] = 10;
    RET[programCount - 1] = 2;
}

