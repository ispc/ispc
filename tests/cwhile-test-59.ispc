export void f_f(uniform float RET[], uniform float vFOO[]) {
    float v = vFOO[programIndex ];
    float r = 0.;
    cwhile (v > 0.) {
        r += 1.;
        v -= .125;
    }
    RET[programIndex] = r;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 8 + 8 * programIndex;
}
