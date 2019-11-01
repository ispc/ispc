export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int16 a_max = 32767, a_min = -32768; // max and min signed int16
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
        RET[programIndex] = (uniform int16) 32767;
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform int16) -32768;
    }
    else {
        RET[programIndex] = (uniform int16) -32763;
    } 
}
