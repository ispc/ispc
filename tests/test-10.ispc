export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float r = a;
    if (a <= 2.) r = b;
    RET[programIndex] = r;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
    RET[0] = RET[1] = 5;
}
