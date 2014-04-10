
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int64 b_conv = b;
    if (programIndex % 3 == 0) {
        uniform saturated int8 tmp1 = 0x7F; // max signed int8
        tmp1 *= b_conv;
        RET[programIndex] = tmp1;
    }
    else if (programIndex % 3 == 1) {
        uniform saturated int8 tmp2 = -0x80; // min signed int8
        tmp2 *= b_conv;
        RET[programIndex] = tmp2;
    }
    else {
        uniform saturated int8 c = b;
        c *= c;
        RET[programIndex] = c;
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform saturated int8) 0x7F; // max signed int8
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform saturated int8) -0x80; // min signed int8
    }
    else {
        RET[programIndex] = (uniform saturated int8) 25;
    } 
}
