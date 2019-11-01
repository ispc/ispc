// rule: skip on arch=genx32
// rule: skip on arch=genx64

static uniform float array[10000];
task void x(uniform int i, float f) {
    uniform int j;
    array[i] = i / 10000.;
    #pragma ignore warning
    cfor (j = 0; j < 10000; ++j)
        array[i] = sin(array[i]);
    if (array[i] < .02)
        array[i] = i;
}
export void f_f(uniform float RET[], uniform float fFOO[]) { 
    float f = fFOO[programIndex];
    uniform int i;
    #pragma ignore warning
    cfor (i = 0; i < 10000; ++i)
        launch x(i, f);
    sync;
    RET[programIndex] = array[9999];
}


export void result(uniform float RET[]) {
    RET[programIndex] = 9999.000000;
}
