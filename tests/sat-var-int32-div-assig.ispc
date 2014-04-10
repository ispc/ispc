

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int64 b_conv = b;
    if (programIndex % 3 == 0) {
        varying saturated int32 tmp1 = 0x7FFFFFFF; // max signed int32
        tmp1 /= b_conv;
        RET[programIndex] = tmp1;
    }
    else if (programIndex % 3 == 1) {
        varying saturated int32 tmp2 = -0x80000000; // min signed int32
        tmp2 /= b_conv;
        RET[programIndex] = tmp2;
    }
    else {
        varying saturated int32 tmp3 = -0x80000000; // min signed int32
        tmp3 /= (-1);
        RET[programIndex] = tmp3;
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
