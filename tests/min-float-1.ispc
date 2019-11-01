export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = min(3 * a, 10.f);
    RET[programCount-1] = min(b, 100.f);
}


export void result(uniform float RET[]) {
    RET[programIndex] = 10;
    RET[0] = 3;
    RET[1] = 6;
    RET[2] = 9;
    RET[programCount-1] = 5;
}
