export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float i, j;
    for (i = 0; i < b; ++i) {
        cif (a == 1.) break;
        for (j = 0; j < b; ++j) {
            cif (a == 3.) break;
            ++a;
        }
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 26+programIndex;
    RET[0] = 1;
    RET[1] = RET[2] = 3;
}
