export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int64 a_max = 0xFFFFFFFFFFFFFFFF, a_min = 0; // max and min unsigned int64
    if (programIndex % 3 == 0) {
        RET[programIndex] = saturating_mul(a_max, (uniform unsigned int64) b);
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = saturating_mul(a_min, (uniform unsigned int64) -b);
    }
    else {
        RET[programIndex] = saturating_mul((uniform unsigned int64) b,
                                           (uniform unsigned int64) b);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (uniform unsigned int64) 0xFFFFFFFFFFFFFFFF; // max unsigned int64
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform unsigned int64) 0; // min unsigned int64
    }
    else {
        RET[programIndex] = (uniform unsigned int64) 25;
    } 
}
