int64 zero = 0;
void *gptr;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        a[i] = aFOO[i];
    
    int g = 0;
    int *ptr = (aFOO[0] == 1234) ? (int * varying)gptr : (a + programIndex);
    if (programIndex < 2) {
        #pragma ignore warning(perf)
        g = *ptr;
    }
    RET[programIndex] = g; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = 1;
    RET[1] = 2;
}
