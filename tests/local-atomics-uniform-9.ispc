uniform int64 s = 100.;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    uniform int64 b = atomic_compare_exchange_local(&s, 100, -100);
    RET[programIndex] = s;
}

export void result(uniform float RET[]) {
    RET[programIndex] = -100.;
}
