
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 3 == 0) {
        varying saturated unsigned int8 tmp1 = 0xFF; // max unsigned int8
        tmp1++;
        RET[programIndex] = tmp1;
    }
    else if (programIndex % 3 == 1) {
        varying saturated unsigned int8 tmp2 = 0; // min unsigned int8
        RET[programIndex] = tmp2++;
    }
    else {
        varying saturated unsigned int8 tmp3 = 0; // min unsigned int8
        tmp3++;
        RET[programIndex] = tmp3;
    }   
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (varying saturated unsigned int8) 0xFF; // max unsigned int8
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (varying saturated unsigned int8) 0; // min unsigned int8
    }
    else {
        RET[programIndex] = (varying saturated unsigned int8) 1; // min unsigned int8 + 1
    }
}
