// rule: skip on cpu=TGLLP

export void f_v(uniform float RET[]) {
#define width 4
//CO    const uniform int width = 4;
    uniform double a[width*programCount], r[width*programCount];
    for (uniform int i = 0; i < width*programCount; ++i)
        a[i] = -1;

    double x = width * programIndex;
    double y = 1 + width * programIndex;
    double z = 2 + width * programIndex;
    double w = 3 + width * programIndex;

    soa_to_aos4(x, y, z, w, a);

    uniform int errs = 0;
    for (uniform int i = 0; i < width * programCount; ++i)
        if (a[i] != i) ++errs;

    RET[programIndex] = errs;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}

