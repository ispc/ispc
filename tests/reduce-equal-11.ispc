// rule: skip on cpu=TGLLP

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    double a = aFOO[programIndex];
    RET[programIndex] = 0;
    if (programIndex & 1)
        RET[programIndex] = reduce_equal(a) ? 1 : 0; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
