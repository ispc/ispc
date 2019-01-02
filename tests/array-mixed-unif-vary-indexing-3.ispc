
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    assert(programCount <= 64);
#ifdef __NVPTX__
    uniform float  * uniform xarr   =  uniform new uniform float[70*70];
    uniform float (* uniform x)[70] = (uniform float (* uniform)[70])xarr;
#define _SHMALLOC
#else
    uniform float x[70][70];
#endif
    for (uniform int i = 0; i < 70; ++i)
        for (uniform int j = 0; j < 70; ++j)
            x[i][j] = 2+b-5;

    // all are 2 except (4,2) = 0, (4,...) = 1, (4,programCount-1)=2
    if (a == 3.) {
        #pragma ignore warning(perf)
        x[b-1][a-1] = 0;
    }
    else {
        #pragma ignore warning(perf)
        x[b-1][a-1] = 1;
    }
    #pragma ignore warning(perf)
    RET[programIndex] = x[4][a];

#ifdef _SHMALLOC
    delete xarr;
#endif
}

export void result(uniform float RET[]) { 
    RET[programIndex] = 1;
    RET[1] = 0;
    RET[programCount-1] = 2;
}
