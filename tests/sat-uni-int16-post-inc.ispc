

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 3 == 0) {
        uniform saturated int16 tmp1 = 0x7FFF; // max signed int16
        tmp1++;
        RET[programIndex] = tmp1;
    }
    else if (programIndex % 3 == 1) {
        uniform saturated int16 tmp2 = -0x8000; // min signed int16
        RET[programIndex] = tmp2++;
    }
    else {
        uniform saturated int16 tmp3 = -0x8000; // min signed int16
        tmp3++;
        RET[programIndex] = tmp3;
    }   
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform saturated int16) 0x7FFF; // max signed int16
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform saturated int16) -0x8000; // min signed int16
    }
    else {
        RET[programIndex] = (uniform saturated int16) -0x7FFF; // min signed int16 + 1
    }
}
