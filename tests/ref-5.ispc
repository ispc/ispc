void foo(float & a) {
    a = 0;
}

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float x[10];
    uniform int i;
    for (i = 0; i < 10; ++i)
        x[i] = a*b;
    foo(x[b]);
    RET[programIndex] = x[5] + x[9];
}

export void result(uniform float RET[]) { RET[programIndex] = 5 * (programIndex+1); }
