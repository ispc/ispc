export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = extract(a, min(1, programCount-1));
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programCount == 1) ? 1 : 2.;
}
