uniform float s = 100.;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    uniform float b = atomic_compare_exchange_global(&s, 1., -100.);
    RET[programIndex] = b;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 100.;
}
