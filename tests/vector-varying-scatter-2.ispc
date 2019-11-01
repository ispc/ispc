export void f_f(uniform float RET[], uniform float aFOO[]) {
    float aa = aFOO[programIndex]; 
    float<4> a = { -1, -2, -3, -4 };
    if (programIndex < 4) {
        // TODO: we need to optimize ispc to not throw out-of-bound warning here.
        #pragma ignore warning
        a[3-programIndex] = aa;
    }
//CO    print("%\n%\n%\n%\n", a[0], a[1], a[2], a[3]);
    int i = clamp(3-programIndex, 0, 3);
//CO    print("%\n%\n", i, a[i]);
    #pragma ignore warning(perf)
    RET[programIndex] = a[i];
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programIndex < 4) ? 1+programIndex : -1;
}
