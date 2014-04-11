
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 2 == 0) {
        uniform saturated int64 tmp1 = 0x7FFFFFFFFFFFFFFF; // max signed int64
        RET[programIndex] = ++tmp1;
    }
    else {
        uniform saturated int64 tmp2 = -0x8000000000000000; // min signed int64
        RET[programIndex] = ++tmp2;
    }  
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (uniform saturated int64) 0x7FFFFFFFFFFFFFFF; // max signed int64
    }
    else {
        RET[programIndex] = (uniform saturated int64) -0x7FFFFFFFFFFFFFFF; // min signed int64 + 1
    }
}
