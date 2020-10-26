uniform float s = 100.;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    uniform float b = atomic_swap_global(&s, 1.);
    RET[programIndex] = s;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1.;
}
