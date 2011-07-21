
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int8 a = aFOO[programIndex]; 
    int8 br = broadcast(a, (uniform int)b-2);
    RET[programIndex] = br;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 4;
}
