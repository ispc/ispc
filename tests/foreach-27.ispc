uniform int foo(int i);

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int x[21];
    for (uniform int i = 0; i < 21; ++i)
        x[i] = i;

    foreach (i = 0 ... 21) {
        if (x[i] == 12345)
            continue;
        x[i] = 1;
    }

    #pragma ignore warning(perf)
    RET[programIndex] = x[min(programIndex, 20)];
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
