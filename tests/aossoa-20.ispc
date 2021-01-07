// rule: skip on cpu=TGLLP

export void f_v(uniform float RET[]) {
#define width 2ul
#define maxProgramCount 64ul
    assert(programCount <= maxProgramCount);

//CO    const uniform int width = 2;
//CO    const uniform int maxProgramCount = 16;
    uniform int64 a[width*maxProgramCount], r[width*maxProgramCount];
    for (uniform int i = 0; i < width*maxProgramCount; ++i)
        a[i] = i;

    int64 x=-1, y=-1;
    aos_to_soa2(a, &x, &y);

    int errs = 0;
    if (x != width * programIndex) ++errs;
    if (y != 1 + width * programIndex) ++errs;

    RET[programIndex] = errs;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}

