export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[programCount+15];
    uniform int i;
    for (i = 0; i < programCount+15; ++i)
        x[i] = i;

    float ret;
    if ((int)a & 1) {
        #pragma ignore warning(perf)
        ret = x[a-1];
    }
    else {
        #pragma ignore warning(perf) 
        ret = x[a];
    }
    RET[programIndex] = ret;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 2 * ((programIndex+1) / 2);
}
