
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform saturated int8 a_max = 0x7F, a_min = -0x80; // max and min signed int8
    if (programIndex % 3 == 0) {
        RET[programIndex] = a_max + b;
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (-b) + a_min;
    }
    else {
        RET[programIndex] = a_min + b;
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform saturated int8) 0x7F; // max signed int8
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform saturated int8) -0x80; // min signed int8
    }
    else {
        RET[programIndex] = (uniform saturated int8) -0x7B; // min + 5
    } 
}
