export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = min(-10 * (a-3), .1f);
    RET[programCount-1] = min(-10 * b, 2.f);
}

export void result(uniform float RET[]) {
    RET[programIndex] = -10 * (programIndex - 2);
    RET[0] = .1;
    RET[1] = .1;
    RET[programCount - 1] = -50;
}

