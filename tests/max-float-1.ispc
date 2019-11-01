export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = max(3 * a, 10.f);
    RET[programCount-1] = max(b, 100.f);
}


export void result(uniform float RET[]) {
    RET[programIndex] = 3 * (1+programIndex);
    RET[0] = 10;
    RET[1] = 10;
    RET[2] = 10;
    RET[programCount-1] = 100;
}
