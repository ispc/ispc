export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying unsigned int8 a_max = 0xFF, a_min = 0; // max and min unsigned int8
    if (programIndex % 2 == 0) {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_div(a_min, (varying unsigned int8) b);
    }
    else {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_div(a_max, (varying unsigned int8) b);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (varying unsigned int8) 0; // min unsigned int8
    }
    else {
        RET[programIndex] = (varying unsigned int8) 0x33; // max unsigned int8 / 5
    } 
}
