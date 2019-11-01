// rule: skip on arch=genx32
// rule: skip on arch=genx64

export void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 *src = new int32[1024];
    int32 *dst = new int32[1024];

    for (uniform int i = 0; i < 1024; ++i) {
        #pragma ignore warning(perf)
        src[i] = programIndex * 10000 + i;
    }

    if (programIndex == 2)
        memcpy(dst, src, programCount*sizeof(uniform int));
    else
        memcpy(dst, src, programCount*sizeof(uniform int));

    #pragma ignore warning(perf)
    RET[programIndex] = dst[programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = 10000 * programIndex + programIndex;
}
