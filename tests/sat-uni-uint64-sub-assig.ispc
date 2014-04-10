

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 2 == 0) {
        uniform saturated unsigned int64 tmp1 = 0; // min unsigned int64
        tmp1 -= b;
        RET[programIndex] = tmp1;
    }
    else {
        uniform saturated unsigned int64 tmp2 = 0xFFFFFFFFFFFFFFFF; // max unsigned int64
        tmp2 -= b;
        RET[programIndex] = tmp2;
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (uniform saturated unsigned int64) 0; // min unsigned int64
    }
    else {
        RET[programIndex] = (uniform saturated unsigned int64) 0xFFFFFFFFFFFFFFFA; // max unsigned int64 - 5
    } 
}
