
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex&3];
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
    RET[0] = RET[4] = RET[8] = RET[12] = 1;
    RET[1] = RET[5] = RET[9] = RET[13] = 3;
    RET[2] = RET[6] = RET[10] = RET[14] = 3;
    RET[3] = RET[7] = RET[11] = RET[15] = 29;
}
