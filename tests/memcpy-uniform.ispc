// rule: skip on arch=genx32
// rule: skip on arch=genx64

export void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 * uniform src = uniform new int32[1024];
    int32 * uniform dst = uniform new int32[1024];

    foreach (i = 0 ... 1024)
        src[i] = i;

    memcpy(&dst[32], src, (1024-32)*sizeof(uniform int));
    RET[programIndex] = dst[64+programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = 32 + programIndex;
}
