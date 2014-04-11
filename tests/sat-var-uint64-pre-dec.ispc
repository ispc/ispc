

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 2 == 0) {
        varying saturated unsigned int64 tmp1 = 0; // min unsigned int64
        RET[programIndex] = --tmp1;
    }
    else {
        varying saturated unsigned int64 tmp2 = 0xFFFFFFFFFFFFFFFF; // max unsigned int64
        RET[programIndex] = --tmp2;
    }  
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (varying saturated unsigned int64) 0; // min unsigned int64
    }
    else {
        RET[programIndex] = (varying saturated unsigned int64) 0xFFFFFFFFFFFFFFFE; // max unsigned int64 - 1
    }
}
