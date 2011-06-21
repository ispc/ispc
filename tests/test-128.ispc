
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = extract(a, 1);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2.;
}
