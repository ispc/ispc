export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform int x[7];
    for (uniform int i = 0; i < 7; ++i)
        x[i] = b;
    RET[programIndex] = x[b];
}


export void result(uniform float RET[]) {
    RET[programIndex] = 5.000000;
}
