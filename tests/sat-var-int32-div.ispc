

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying saturated int32 a_max = 0x7FFFFFFF, a_min = -0x80000000; // max and min signed int32
    uniform unsigned int64 b_conv = b;
    if (programIndex % 3 == 0) {
        RET[programIndex] = a_max / b_conv;
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = a_min / b_conv;
    }
    else {
        RET[programIndex] = a_min / (-1);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (varying saturated int32) 429496729.4; // max / 5
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (varying saturated int32) -429496729.6; // min / 5 
    }
    else {
        RET[programIndex] = (varying saturated int32) 0x7FFFFFFF; // max signed int32
    } 
}
