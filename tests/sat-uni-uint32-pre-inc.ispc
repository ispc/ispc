
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 2 == 0) {
        uniform saturated unsigned int32 tmp1 = 0xFFFFFFFF; // max unsigned int32
        RET[programIndex] = ++tmp1;
    }
    else {
        uniform saturated unsigned int32 tmp2 = 0; // min unsigned int32
        RET[programIndex] = ++tmp2;
    }  
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (uniform saturated unsigned int32) 0xFFFFFFFF; // max unsigned int32
    }
    else {
        RET[programIndex] = (uniform saturated unsigned int32) 1; // min unsigned int32 + 1
    }
}
