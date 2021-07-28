#include "../test_static.isph"
task void f_v(uniform float RET[]) {
#define width 2ul
#define maxProgramCount 64ul
    assert(programCount <= maxProgramCount);

//CO    const uniform int width = 2;
//CO    const uniform int maxProgramCount = 16;
    uniform int a[width*maxProgramCount], r[width*maxProgramCount];
    for (uniform int i = 0; i < width*maxProgramCount; ++i)
        a[i] = i;

    int x=-1, y=-1;
    aos_to_soa2(a, &x, &y);

    int errs = 0;
    if (x != width * programIndex) ++errs;
    if (y != 1 + width * programIndex) ++errs;

    RET[programIndex] = errs;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
