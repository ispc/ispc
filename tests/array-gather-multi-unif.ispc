
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex];
    uniform float x[34][34];
    // HACK to prevent the init from getting turned into a @llvm.memset
    // intrinsic, which the JIT doesn't deal with...
    for (uniform int i = 0; i < 29+b; ++i)
        for (uniform int j = 0; j < 29+b; ++j)
            x[i][j] = 0;
    if (a < 34) {
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
