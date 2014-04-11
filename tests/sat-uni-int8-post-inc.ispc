
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 3 == 0) {
        uniform saturated int8 tmp1 = 0x7F; // max signed int8
        tmp1++;
        RET[programIndex] = tmp1;
    }
    else if (programIndex % 3 == 1) {
        uniform saturated int8 tmp2 = -0x80; // min signed int8
        RET[programIndex] = tmp2++;
    }
    else {
        uniform saturated int8 tmp3 = -0x80; // min signed int8
        tmp3++;
        RET[programIndex] = tmp3;
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
        RET[programIndex] = (uniform saturated int8) -0x7F; // min signed int8 + 1
    }
}
