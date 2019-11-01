export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programCount == 1)
        RET[0] = 3;
    else {
        float aa = aFOO[programIndex]; 
        float bb = aa + programCount;
        float shuf = shuffle(aa, bb, programIndex + 2 + (int)b - 5);
        RET[programIndex] = shuf;
    }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 3 + programIndex;
}
