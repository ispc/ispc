export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    int i = (int)a;
    RET[programIndex] = max((int)2, i);
    RET[programCount-1] = max(10, (int)b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex + 1;
    RET[0] = 2;
    RET[programCount-1] = 10;
}

