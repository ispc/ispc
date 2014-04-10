
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying saturated unsigned int8 a_max = 0xFF, a_min = 0; // max and min unsigned int8
    if (programIndex % 2 == 0) {
        RET[programIndex] = a_min - b;
    }
    else {
        RET[programIndex] = a_max - b;
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (varying saturated unsigned int8) 0; // min unsigned int8
    }
    else {
        RET[programIndex] = (varying saturated unsigned int8) 0xFA; // max unsigned int8 - 5
    } 
}
