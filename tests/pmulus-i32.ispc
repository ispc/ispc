export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int32 a_max = 0xFFFFFFFF, a_min = 0; // max and min unsigned int32
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_mul(a_max, (uniform unsigned int32) b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_mul(a_min, (uniform unsigned int32) -b);
    }
    else {
        RET[programIndex] = saturating_mul((uniform unsigned int32) b,
                                           (uniform unsigned int32) b);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform unsigned int32) 0xFFFFFFFF; // max unsigned int32
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform unsigned int32) 0; // min unsigned int32
    }
    else {
        RET[programIndex] = (uniform unsigned int32) 25;
    } 
}
