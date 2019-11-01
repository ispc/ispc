export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float r;
    b -= 2.;
    RET[programIndex] = (a == b) ? a : 2.*a;
}


export void result(uniform float RET[4]) {
    RET[programIndex] = 2 + 2*programIndex;
    RET[2] = 3;
}
