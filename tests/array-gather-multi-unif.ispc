// We have all checks in place, so we are safe and don't read or write beyond the boundaries.
#define SIZE 34

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex];
    uniform float x[SIZE][SIZE];
    // HACK to prevent the init from getting turned into a @llvm.memset
    // intrinsic, which the JIT doesn't deal with...
    for (uniform int i = 0; i < SIZE-5+b; ++i)
        for (uniform int j = 0; j < SIZE-5+b; ++j)
            x[i][j] = 0;
    if (a < SIZE) {
        #pragma ignore warning(perf)
        x[a][a] = a;
    }
    RET[programIndex] = x[4][4] + x[1][1] + x[b][b] + x[0][0];
}

export void result(uniform float RET[]) {
    if (programCount == 1)
        RET[programIndex] = 1;
    else if (programCount == 4)
        RET[programIndex] = 5.;
    else
        RET[programIndex] = 10.;
}
