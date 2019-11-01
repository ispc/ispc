export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex]; 
    int br = (programCount == 1) ? 4 : broadcast(a, (uniform int)b-2);
    RET[programIndex] = br;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 4;
}
