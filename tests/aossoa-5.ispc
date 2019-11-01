export void f_v(uniform float RET[]) {
#define width 3ul
#define maxProgramCount 64ul
    assert(programCount <= maxProgramCount);

//CO    const uniform int width = 3;
//CO    const uniform int maxProgramCount = 16;
    uniform int a[width*maxProgramCount], r[width*maxProgramCount];
    for (uniform int i = 0; i < width*maxProgramCount; ++i)
        a[i] = i;

    int x=-1, y=-1, z=-1;
    aos_to_soa3(a, &x, &y, &z);

    int errs = 0;
    if (x != width * programIndex) ++errs;
    if (y != 1 + width * programIndex) ++errs;
    if (z != 2 + width * programIndex) ++errs;

    RET[programIndex] = errs; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}

