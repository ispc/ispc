
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex&0x3];
    uniform float i;
    cfor (i = 0; i < b; ++i) {
        if (i == 2) cbreak;
        ++a;
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    uniform float ret[4] = { 3, 4, 5, 6 };
    RET[programIndex] = ret[programIndex & 0x3];
}
