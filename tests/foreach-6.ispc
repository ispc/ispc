uniform int foo(int i);

export void f_f(uniform float RET[], uniform float aFOO[]) {
    for (uniform int i = 0; i < programCount; ++i)
        RET[i] = 0;

    foreach (i = -2 ... programCount-2)
        RET[i+2] += 1234;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1234;
}
