export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int64 a_max = 0xFFFFFFFFFFFFFFFF, a_min = 0; // max and min unsigned int64
    if (programIndex % 2 == 0) {
        RET[programIndex] = saturating_add(a_max, b);
    }
    else {
        RET[programIndex] = saturating_add(a_min, b);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (uniform unsigned int64) 0xFFFFFFFFFFFFFFFF; // max unsigned int64
    }
    else {
        RET[programIndex] = (uniform unsigned int64) 5;
    } 
}
