export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform int i;
    float x[10];
    #pragma ignore warning
    cfor (i = 0; i < 10; ++i)
        x[i] = i;
    RET[programIndex] = x[b];
}


export void result(uniform float RET[]) {
    RET[programIndex] = 5;
}
