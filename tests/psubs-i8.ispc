export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int8 a_max = 127, a_min = -128; // max and min signed int8
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_sub(a_min, b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_sub(a_max, -b);
    }
    else {
        RET[programIndex] = saturating_sub(a_max, b);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform int8) -128;
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform int8) 127;
    }
    else {
        RET[programIndex] = (uniform int8) 122;
    } 
}
