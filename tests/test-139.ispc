
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex&0x3]; 
    RET[programIndex] = exp(a); 
}

export void result(uniform float RET[]) {
    RET[0] = RET[4] = RET[8] = RET[12] = 0x1.5bf0a8p+1;
    RET[1] = RET[5] = RET[9] = RET[13] = 0x1.d8e64cp+2;
    RET[2] = RET[6] = RET[10] = RET[14] = 0x1.415e5cp+4;
    RET[3] = RET[7] = RET[11] = RET[15] = 0x1.b4c904p+5;
}
