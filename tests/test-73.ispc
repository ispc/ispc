export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float i, r = 0.;
    for (i = 0; i < a; ++i)
        r += b;
    RET[programIndex] = r;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 5 + 5 * programIndex;
}
