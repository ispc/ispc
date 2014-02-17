
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int32 a_max = 2147483647, a_min = -2147483648; // max and min int32
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
        RET[programIndex] = (uniform int32) -2147483648;
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform int32) 2147483647;
    }
    else {
        RET[programIndex] = (uniform int32) 2147483642;
    } 
}
