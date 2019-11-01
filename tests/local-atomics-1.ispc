uniform unsigned int32 s = 0;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float delta = 1;
    #pragma ignore warning(perf)
    float b = atomic_add_local(&s, delta);
    RET[programIndex] = reduce_add(b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = reduce_add(programIndex);
}
