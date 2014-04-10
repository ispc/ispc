
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 2 == 0) {
        uniform saturated unsigned int8 tmp1 = 0xFF; // max unsigned int8
        tmp1 += b;
        RET[programIndex] = tmp1;
    }
    else {
        uniform saturated unsigned int8 tmp2 = 0; // min unsigned int8
        tmp2 += b;
        RET[programIndex] = tmp2;
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (uniform saturated unsigned int8) 0xFF; // max unsigned int8
    }
    else {
        RET[programIndex] = (uniform saturated unsigned int8) 5;
    } 
}
