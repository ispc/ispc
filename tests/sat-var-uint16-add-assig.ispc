
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 2 == 0) {
        varying saturated unsigned int16 tmp1 = 0xFFFF; // max unsigned int16
        tmp1 += b;
        RET[programIndex] = tmp1;
    }
    else {
        varying saturated unsigned int16 tmp2 = 0; // min unsigned int16
        tmp2 += b;
        RET[programIndex] = tmp2;
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (varying saturated unsigned int16) 0xFFFF; // max unsigned int16
    }
    else {
        RET[programIndex] = (varying saturated unsigned int16) 5;
    } 
}
