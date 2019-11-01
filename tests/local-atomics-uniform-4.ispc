uniform unsigned int32 s = 0xffff;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    uniform unsigned int32 b = atomic_min_local(&s, 1);
    RET[programIndex] = b;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0xffff;
}
