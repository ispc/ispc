export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying unsigned int16 a_max = 0xFFFF, a_min = 0; // max and min unsigned int16
    if (programIndex % 2 == 0) {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_div(a_min, (varying unsigned int16) b);
    }
    else {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_div(a_max, (varying unsigned int16) b);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (varying unsigned int16) 0; // min unsigned int16
    }
    else {
        RET[programIndex] = (varying unsigned int16) 0x3333; // max unsigned int16 / 5
    } 
}
