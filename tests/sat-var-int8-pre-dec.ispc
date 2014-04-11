
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 2 == 0) {
        varying saturated int8 tmp1 = -0x80; // min signed int8
        RET[programIndex] = --tmp1;
    }
    else {
        varying saturated int8 tmp2 = 0x7F; // max signed int8
        RET[programIndex] = --tmp2;
    }  
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (varying saturated int8) -0x80; // min signed int8
    }
    else {
        RET[programIndex] = (varying saturated int8) 0x7E; // max signed int8 - 1
    }
}
