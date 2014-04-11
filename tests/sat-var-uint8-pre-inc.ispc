
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 2 == 0) {
        varying saturated unsigned int8 tmp1 = 0xFF; // max unsigned int8
        RET[programIndex] = ++tmp1;
    }
    else {
        varying saturated unsigned int8 tmp2 = 0; // min unsigned int8
        RET[programIndex] = ++tmp2;
    }  
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (varying saturated unsigned int8) 0xFF; // max unsigned int8
    }
    else {
        RET[programIndex] = (varying saturated unsigned int8) 1; // min unsigned int8 + 1
    }
}
