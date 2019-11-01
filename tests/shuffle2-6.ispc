export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int8 aa = aFOO[programIndex]; 
    int8 bb = aa + programCount;
    int8 shuf = shuffle(aa, bb, 1);
    RET[programIndex] = shuf;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2;
}
