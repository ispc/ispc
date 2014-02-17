
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying int8 a_max = 127, a_min = -128; // max and min signed int8
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_div(a_max, (varying int8) b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_div(a_min, (varying int8) b);
    }
    else {
        RET[programIndex] = saturating_div(a_min, (varying int8) -1);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (varying int8) 25.4;
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (varying int8) -25.6;
    }
    else {
        RET[programIndex] = (varying int8) 127;
    } 
}
