export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    a -= 2;
    if (a > 0)
        // This shoudln't hit, given the if test above
        assert(a > 0);
    assert(b == 5.);
    RET[programIndex] = a+2; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
