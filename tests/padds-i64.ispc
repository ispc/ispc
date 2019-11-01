export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int64 a_max = 0x7FFFFFFFFFFFFFFF, a_min = 0x8000000000000000; // max and min signed int64
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
        RET[programIndex] = (uniform int64) 0x7FFFFFFFFFFFFFFF; // max signed int64
    }
    else if (programIndex % 3 == 1) {
        RET[programIndex] = (uniform int64) 0x8000000000000000; // min signed int64
    }
    else {
        RET[programIndex] = (uniform int64) -0x7FFFFFFFFFFFFFFB; // min + 5
    } 
}
