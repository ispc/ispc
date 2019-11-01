export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    varying unsigned int8 a_max = 255, a_min = 0; // max and min signed int8
    if (programIndex % 2 == 0) {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_add(a_max, b);
    }
    else {
        #pragma ignore warning(perf)
        RET[programIndex] = saturating_add(a_min, b);
    } 
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (varying unsigned int8) 255;
    }
    else {
        RET[programIndex] = (varying unsigned int8) 5;
    } 
}

