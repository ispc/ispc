export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a[programCount];
    a[programIndex] = aFOO[programIndex];
    int aa;
    uniform int count = 0;
    if (programIndex < 2)
        count += packed_load_active(a, &aa);
    RET[programIndex] = count;
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programCount == 1) ? 1 : 2;
}
