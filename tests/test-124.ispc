export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    if (any(a == 3)) { RET[programIndex] = 1; return; }
    RET[programIndex] = 0;
}


export void result(uniform float RET[4]) {
    RET[programIndex] = (programCount == 1) ? 0 : 1;
}
