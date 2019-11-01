export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int32 a_max = 0xFFFFFFFF, a_min = 0; // max and min unsigned int32
    if (programIndex % 2 == 0) {
        RET[programIndex] = saturating_sub(a_min, b);
    }
    else {
        RET[programIndex] = saturating_sub(a_max, b);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (uniform unsigned int32) 0; // min unsigned int32
    }
    else {
        RET[programIndex] = (uniform unsigned int32) 0xFFFFFFFB; // max unsigned int32 - 5
    } 
}
