
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int16 aa = aFOO[programIndex]; 
    int16 bb = aa + programCount;
    int16 shuf = shuffle(aa, bb, programIndex + 2 + (int)b - 5);
    RET[programIndex] = shuf;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 3 + programIndex;
}
