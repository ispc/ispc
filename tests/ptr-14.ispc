export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    varying float * uniform pa = &a;
    varying int * uniform pb = (varying int *)pa;
    varying float *uniform pc = (varying float *)pb;
    *pc = programIndex;
    RET[programIndex] = *pc;
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
