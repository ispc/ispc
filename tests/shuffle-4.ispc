export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int16 a = aFOO[programIndex]; 
    int reverse = programCount - 1 - programIndex;
    int16 shuf = shuffle(a, reverse);
    RET[programIndex] = shuf;
}

export void result(uniform float RET[]) {
    RET[programIndex] = programCount - programIndex;
}
