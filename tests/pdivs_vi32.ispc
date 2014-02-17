
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying int32 a_max = 2147483647, a_min = -2147483648; // max and min signed int32
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_div(a_max, (varying int32) b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_div(a_min, (varying int32) b);
    }
    else {
        RET[programIndex] = saturating_div(a_min, (varying int32) -1);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (varying int32) 429496729.4;
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (varying int32) -429496729.6;
    }
    else {
        RET[programIndex] = (varying int32) 2147483647;
    } 
}
