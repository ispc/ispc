
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying saturated int8 a_max = 0x7F, a_min = -0x80; // max and min signed int8
    uniform unsigned int64 b_conv = b;
    if (programIndex % 3 == 0) {
        RET[programIndex] = a_max * b_conv;
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = a_min * b_conv;
    }
    else {
        varying saturated int8 c = b;
        RET[programIndex] = c * c;
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (varying saturated int8) 0x7F; // max signed int8
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (varying saturated int8) -0x80; // min signed int8
    }
    else {
        RET[programIndex] = (varying saturated int8) 25;
    } 
}
