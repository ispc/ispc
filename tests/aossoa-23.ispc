#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_v(uniform float RET[]) {
#define width 2
//CO    const uniform int width = 2;
    uniform double a[width*programCount], r[width*programCount];
    for (uniform int i = 0; i < width*programCount; ++i)
        a[i] = -1;

    double x = width * programIndex;
    double y = 1 + width * programIndex;

    soa_to_aos2(x, y, a);
    uniform int errs = 0;
    for (uniform int i = 0; i < width * programCount; ++i)
        if (a[i] != i) ++errs;

    RET[programIndex] = errs;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
