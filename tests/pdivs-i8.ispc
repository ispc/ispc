export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int8 a_max = 0x7F, a_min = -0x80; // max and min signed int8
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_div(a_max, (uniform int8) b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_div(a_min, (uniform int8) b);
    }
    else {
        RET[programIndex] = saturating_div(a_min, (uniform int8) -1);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform int8) 25.4; // max / 5
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform int8) -25.6; // min / 5 
    }
    else {
        RET[programIndex] = (uniform int8) 0x7F; // max signed int8
    } 
}
