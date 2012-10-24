
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[4], uniform float aFOO[4], uniform float b) {
    float a = aFOO[programIndex&0x3];
    float i, j;
    for (i = 0; i < b; ++i) {
        if (a == 1.) cbreak;
        for (j = 0; j < b; ++j) {
            if (a == 3.) cbreak;
            ++a;
        }
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 3;
    RET[0] = RET[4] = RET[8] = RET[12] = 1;
    RET[3] = RET[7] = RET[11] = RET[15] = 29;
}
