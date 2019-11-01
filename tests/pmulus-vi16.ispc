export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying unsigned int16 a_max = 0xFFFF, a_min = 0; // max and min unsigned int16
    if (programIndex % 3 == 0) {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_mul(a_max, (varying unsigned int16) b);
    }
    else if (programIndex % 3 == 1) {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_mul(a_min, (varying unsigned int16) -b);
    }
    else {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_mul((varying unsigned int16) b, (varying unsigned int16) b);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 3 == 0) {
        RET[programIndex] = (varying unsigned int16) 0xFFFF; // max unsigned int16
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (varying unsigned int16) 0; // min unsigned int16
    }
    else {
        RET[programIndex] = (varying unsigned int16) 25;
    } 
}
