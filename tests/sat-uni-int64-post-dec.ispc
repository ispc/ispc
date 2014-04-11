

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 3 == 0) {
        uniform saturated int64 tmp1 = -0x8000000000000000; // min signed int64
        tmp1--;
        RET[programIndex] = tmp1;
    }
    else if (programIndex % 3 == 1) {
        uniform saturated int64 tmp2 = 0x7FFFFFFFFFFFFFFF; // max signed int64
        RET[programIndex] = tmp2--;
    }
    else {
        uniform saturated int64 tmp3 = 0x7FFFFFFFFFFFFFFF; // max signed int64
        tmp3--;
        RET[programIndex] = tmp3;
    }   
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform saturated int64) -0x8000000000000000; // min signed int64
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform saturated int64) 0x7FFFFFFFFFFFFFFF; // max signed int64
    }
    else {
        RET[programIndex] = (uniform saturated int64) 0x7FFFFFFFFFFFFFFE; // max signed int64 - 1
    }
}
