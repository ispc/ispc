
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 2 == 0) {
        uniform saturated int16 tmp1 = 0x7FFF; // max signed int16
        RET[programIndex] = ++tmp1;
    }
    else {
        uniform saturated int16 tmp2 = -0x8000; // min signed int16
        RET[programIndex] = ++tmp2;
    }  
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (uniform saturated int16) 0x7FFF; // max signed int16
    }
    else {
        RET[programIndex] = (uniform saturated int16) -0x7FFF; // min signed int16 + 1
    }
}
