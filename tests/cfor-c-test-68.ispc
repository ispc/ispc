
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex&0x3];
    float i, j;
    cfor (i = 0; i < b; ++i) {
        if (a == b) ccontinue;
        ++a;
        if (a == 2) cbreak;
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    uniform float ret[4] = { 2, 5, 5, 5 };
    RET[programIndex] = ret[programIndex & 0x3];
}
