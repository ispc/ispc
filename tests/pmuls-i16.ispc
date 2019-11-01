export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int16 a_max = 0x7FFF, a_min = -0x8000; // max and min signed int16
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_mul(a_max, (uniform int16) b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_mul(a_min, (uniform int16) b);
    }
    else {
        RET[programIndex] = saturating_mul((uniform int16) b,
                                           (uniform int16) b);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform int16) 0x7FFF; // max signed int16
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform int16) -0x8000; // min signed int16
    }
    else {
        RET[programIndex] = (uniform int16) 25;
    } 
}
