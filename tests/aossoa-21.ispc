export void f_v(uniform float RET[]) {
#define width 2
//CO    const uniform int width = 2;
    uniform float a[width*programCount], r[width*programCount];
    for (uniform int i = 0; i < width*programCount; ++i)
        a[i] = -1;

    float x = width * programIndex;
    float y = 1 + width * programIndex;

    soa_to_aos2(x, y, a);
    uniform int errs = 0;
    for (uniform int i = 0; i < width * programCount; ++i)
        if (a[i] != i) ++errs;

    RET[programIndex] = errs;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}

