export void f_f(uniform float RET[], uniform float aFOO[]) {
#define NA 3
#define NB 8
    uniform int a[NA][NB];

    for (uniform int i = 0; i < NA; ++i)
        for (uniform int j = 0; j < NB; ++j)
            a[i][j] = 0;

    foreach_tiled (i = 0 ... NA, j = 0 ... NB) {
        // TODO: Revisit this. Future optimization might remove this warning.
        #pragma ignore warning(perf)
        a[i][j] += 1;
    }

    uniform int errs = 0;
    for (uniform int i = 0; i < NA; ++i)
        for (uniform int j = 0; j < NB; ++j)
            if (a[i][j] != 1) {
                ++errs;
            }

    RET[programIndex] = errs;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
