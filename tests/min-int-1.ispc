export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    int i = (int)a;
    RET[programIndex] = min((int)2, i);
    RET[programCount-1] = min(10, (int)b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2;
    RET[0] = 1;
    RET[programCount-1] = 5;
}

