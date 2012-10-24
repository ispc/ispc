
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex&0x3];
    float i, j;
    cfor (i = 0; i < b; ++i) {
        if (a == 1.) cbreak;
        cfor (j = 0; j < b; ++j) {
            if (a == 3.) cbreak;
            ++a;
        }
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    uniform float ret[4] = { 1, 3, 3, 29 };
    RET[programIndex] = ret[programIndex & 0x3];
}
