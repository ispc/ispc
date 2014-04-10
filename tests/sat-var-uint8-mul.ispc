
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying saturated unsigned int8 a_max = 0xFF, a_min = 0; // max and min unsigned int8
    uniform unsigned int64 b_conv = b;
    if (programIndex % 3 == 0) {
        RET[programIndex] = a_max * b_conv;
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = a_min * (-b_conv);
    }
    else {
        varying saturated unsigned int8 c = b;
        RET[programIndex] = c * c;
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
        RET[programIndex] = (varying saturated unsigned int8) 25;
    } 
}
