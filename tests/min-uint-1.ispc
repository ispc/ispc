export void f_fu(uniform float r[], uniform float a[], uniform float b) {
    #pragma ignore warning(perf)
    unsigned int i = (unsigned int)a[programIndex];
    #pragma ignore warning(perf)
    r[programIndex] = min((varying unsigned int)2, i);
    r[programCount-1] = min((uniform unsigned int)10, (uniform unsigned int)b);
}

export void result(uniform float r[]) { 
    r[programIndex] = 2;
    r[0] = 1;
    r[programCount - 1] = 5;
}


