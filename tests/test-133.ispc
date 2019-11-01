export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float i;
    for (i = 0; i < b; ++i) {
        #pragma ignore warning
        cif (i+2 == b) continue;
        ++a;
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 5 + programIndex;
}
