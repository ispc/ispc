
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying saturated unsigned int16 a_max = 0xFFFF, a_min = 0; // max and min unsigned int16
    if (programIndex % 2 == 0) {
        RET[programIndex] = a_max + b;
    }
    else {
        RET[programIndex] = a_min + b;
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
