
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform int x[7] = b;
    RET[programIndex] = x[b];
}


export void result(uniform float RET[]) {
    RET[programIndex] = 5.000000;
}
