

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 3 == 0) {
        uniform saturated unsigned int64 tmp1 = 0; // min unsigned int64
        tmp1--;
        RET[programIndex] = tmp1;
    }
    else if (programIndex % 3 == 1) {
        uniform saturated unsigned int64 tmp2 = 0xFFFFFFFFFFFFFFFF; // max unsigned int64
        RET[programIndex] = tmp2--;
    }
    else {
        uniform saturated unsigned int64 tmp3 = 0xFFFFFFFFFFFFFFFF; // max unsigned int64
        tmp3--;
        RET[programIndex] = tmp3;
    }   
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform saturated unsigned int64) 0; // min unsigned int64
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform saturated unsigned int64) 0xFFFFFFFFFFFFFFFF; // max unsigned int64
    }
    else {
        RET[programIndex] = (uniform saturated unsigned int64) 0xFFFFFFFFFFFFFFFE; // max unsigned int64 - 1
    }
}
