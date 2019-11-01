export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[45];
    uniform int i;
    #pragma ignore warning
    cfor (i = 0; i < 45; ++i)
        x[i] = i+b;
    a -= 1;
    if (a == 3) a = 0;
    a *= 10000000;
    float ret = 1234;
    if (a < 5) {
        #pragma ignore warning(perf)
        ret = x[a];
    }
    RET[programIndex] = ret;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 1234;
    RET[0] = RET[3] = 5;
}
