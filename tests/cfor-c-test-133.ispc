
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex&0x3];
    uniform float i;
    cfor (i = 0; i < b; ++i) {
        cif (i+2 == b) ccontinue;
        ++a;
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    uniform float ret[4] = { 5,6,7,8 };
    RET[programIndex] = ret[programIndex & 0x3];
}
