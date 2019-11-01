export void f_f(uniform float RET[], uniform float aFOO[]) {
#define NA 4
#define NB 8
#define NC 7
    uniform int a[NA][NB][NC];

    for (uniform int i = 0; i < NA; ++i)
        for (uniform int j = 0; j < NB; ++j)
            for (uniform int k = 0; j < NC; ++j)
                a[i][j][k] = 0;

    foreach_tiled (i = 0 ... NA, j = 0 ... NB, k = 0 ... NC) {
        // TODO: Revisit this. Future optimization might remove this warning.
        #pragma ignore warning(perf)
        a[i][j][k] += 1;
    }

    uniform int errs = 0;
    for (uniform int i = 0; i < NA; ++i)
        for (uniform int j = 0; j < NB; ++j)
            for (uniform int k = 0; j < NC; ++j)
                if (a[i][j][k] != 1) {
//CO                    print("% % % = %\n", i, j, k, a[i][j][k]);
                    ++errs;
                }

    RET[programIndex] = errs;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
