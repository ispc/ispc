
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform saturated int64 a_max = 0x7FFFFFFFFFFFFFFF, a_min = -0x8000000000000000; // max and min signed int64
    uniform unsigned int64 b_conv = b;
    if (programIndex % 3 == 0) {
        RET[programIndex] = a_max * b_conv;
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = a_min * b_conv;
    }
    else {
        uniform saturated int64 c = b;
        RET[programIndex] = c * c;
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform saturated int64) 0x7FFFFFFFFFFFFFFF; // max signed int64
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform saturated int64) -0x8000000000000000; // min signed int64
    }
    else {
        RET[programIndex] = (uniform saturated int64) 25;
    } 
}
