export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float i;
    #pragma ignore warning
    cfor (i = 0; i < b; ++i)
        ++a;
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 6+programIndex;
}
