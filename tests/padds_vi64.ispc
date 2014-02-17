
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying int64 a_max = 9223372036854775807, a_min = -9223372036854775808; // max and min signed int64
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_add(a_max, b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_add(a_min, -b);
    }
    else {
        RET[programIndex] = saturating_add(a_min, b);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (varying int64) 9223372036854775807;
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (varying int64) -9223372036854775808;
    }
    else {
        RET[programIndex] = (varying int64) -9223372036854775803;
    } 
}
