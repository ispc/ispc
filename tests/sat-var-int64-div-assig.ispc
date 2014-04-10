

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int64 b_conv = b;
    if (programIndex % 3 == 0) {
        varying saturated int64 tmp1 = 0x7FFFFFFFFFFFFFFF; // max signed int64
        tmp1 /= b_conv;
        RET[programIndex] = tmp1;
    }
    else if (programIndex % 3 == 1) {
        varying saturated int64 tmp2 = -0x8000000000000000; // min signed int64
        tmp2 /= b_conv;
        RET[programIndex] = tmp2;
    }
    else {
        varying saturated int64 tmp3 = -0x8000000000000000; // min signed int64
        tmp3 /= (-1);
        RET[programIndex] = tmp3;
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (varying saturated int64) 1844674407370955161.4; // max / 5
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (varying saturated int64) -1844674407370955161.6; // min / 5 
    }
    else {
        RET[programIndex] = (varying saturated int64) 0x7FFFFFFFFFFFFFFF; // max signed int64
    } 
}
