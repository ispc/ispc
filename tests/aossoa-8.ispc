export void f_v(uniform float RET[]) {
#define width 3
//CO    const uniform int width = 4;
    uniform int a[width*programCount], r[width*programCount];
    for (uniform int i = 0; i < width*programCount; ++i)
        a[i] = -1;

    int x = width * programIndex;
    int y = 1 + width * programIndex;
    int z = 2 + width * programIndex;

    soa_to_aos3(x, y, z, a);
    uniform int errs = 0;
    for (uniform int i = 0; i < width * programCount; ++i)
        if (a[i] != i) ++errs;

    RET[programIndex] = errs; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}

