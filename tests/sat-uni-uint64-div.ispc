

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform saturated unsigned int64 a_max = 0xFFFFFFFFFFFFFFFF, a_min = 0; // max and min unsigned int64
    uniform unsigned int64 b_conv = b;
    if (programIndex % 2 == 0) {
        RET[programIndex] = a_min / b_conv;
    }
    else {
        RET[programIndex] = a_max / b_conv;
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (uniform saturated unsigned int64) 0; // min unsigned int64
    }
    else {
        RET[programIndex] = (uniform saturated unsigned int64) 0x3333333333333333; // max unsigned int64 / 5
    } 
}
