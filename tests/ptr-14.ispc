
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float * uniform pa = &a;
    int * uniform pb = (int *)pa;
    float *uniform pc = (float *)pb;
    *pc = programIndex;
    RET[programIndex] = *pc;
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
