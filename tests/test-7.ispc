export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = a + 2. * b;
}


export void result(uniform float RET[4]) {
    RET[programIndex] = 11 + programIndex;
}
