export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int64 a[100];
    for (uniform int64 i = 0; i < 100; ++i)
        a[i] = i;
    int64 aa = 32;
    uniform int count = 0;
    if (programIndex < 2)
        count += packed_load_active(&a[5], &aa);
    RET[programIndex] = aa;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 32;
    RET[0] = 5;
    RET[1] = 6;
}
