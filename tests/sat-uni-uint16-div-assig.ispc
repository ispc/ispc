
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int64 b_conv = b;
    if (programIndex % 2 == 0) {
        uniform saturated unsigned int16 tmp1 = 0; // min unsigned int16
        tmp1 /= b_conv;
        RET[programIndex] = tmp1;
    }
    else {
        uniform saturated unsigned int16 tmp2 = 0xFFFF; // max unsigned int16
        tmp2 /= b_conv;
        RET[programIndex] = tmp2;
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (uniform saturated unsigned int16) 0; // min unsigned int16
    }
    else {
        RET[programIndex] = (uniform saturated unsigned int16) 0x3333; // max unsigned int16 / 5
    } 
}
