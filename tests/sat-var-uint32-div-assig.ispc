
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int64 b_conv = b;
    if (programIndex % 2 == 0) {
        varying saturated unsigned int32 tmp1 = 0; // min unsigned int32
        tmp1 /= b_conv;
        RET[programIndex] = tmp1;
    }
    else {
        varying saturated unsigned int32 tmp2 = 0xFFFFFFFF; // max unsigned int32
        tmp2 /= b_conv;
        RET[programIndex] = tmp2;
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (varying saturated unsigned int32) 0; // min unsigned int32
    }
    else {
        RET[programIndex] = (varying saturated unsigned int32) 0x33333333; // max unsigned int32 / 5
    } 
}
