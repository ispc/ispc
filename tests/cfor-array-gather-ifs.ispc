#define SIZE TARGET_WIDTH+1

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[SIZE];
    uniform int i;
    #pragma ignore warning
    cfor (i = 0; i < SIZE; ++i)
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
