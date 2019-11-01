export void f_f(uniform float r[], uniform float a[]) {
    #pragma ignore warning(perf)
    unsigned int i = (unsigned int)a[programIndex];
    #pragma ignore warning(perf)
    r[programIndex] =  min((unsigned int)200, i);
}

export void result(uniform float r[]) { 
    r[programIndex] = 1+programIndex;
}


