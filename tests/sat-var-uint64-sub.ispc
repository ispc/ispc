

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying saturated unsigned int64 a_max = 0xFFFFFFFFFFFFFFFF, a_min = 0; // max and min unsigned int64
    if (programIndex % 2 == 0) {
        RET[programIndex] = a_min - b;
    }
    else {
        RET[programIndex] = a_max - b;
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (varying saturated unsigned int64) 0; // min unsigned int64
    }
    else {
        RET[programIndex] = (varying saturated unsigned int64) 0xFFFFFFFFFFFFFFFA; // max unsigned int64 - 5
    } 
}
