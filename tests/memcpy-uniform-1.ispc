#define N 64
export void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 uniform src[N];
    int32 uniform dst[N];

    foreach (i = 0 ... N) {
        src[i] = i;
        dst[i] = 0;
    }

    memcpy(&dst[1], src, (programCount-2)*sizeof(uniform int));
    RET[programIndex] = dst[programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex-1;
    RET[programCount-1] = 0;
    RET[0] = 0;
}

