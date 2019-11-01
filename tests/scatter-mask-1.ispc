uniform float a[programCount];

export void f_f(uniform float RET[], uniform float aFOO[]) {
    int index = aFOO[programIndex]-1; 
    if (index & 1) {
        #pragma ignore warning(perf)
        a[index] = 1;
    }
    RET[programIndex] = a[programIndex]; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programIndex & 1) ? 1 : 0;
}
