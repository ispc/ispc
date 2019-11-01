export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying unsigned int64 a_max = 0xFFFFFFFFFFFFFFFF, a_min = 0; // max and min unsigned int64
    if (programIndex % 2 == 0) {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_sub(a_min, b);
    }
    else {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_sub(a_max, b);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (varying unsigned int64) 0; // min unsigned int64
    }
    else {
        RET[programIndex] = (varying unsigned int64) 0xFFFFFFFFFFFFFFFB; // max unsigned int64 - 5
    } 
}
