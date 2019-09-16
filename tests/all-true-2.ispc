export void f_f(uniform float RET[], uniform float aFOO[]) {

    int errs = 0;
    int a = aFOO[programIndex];

    for (int i = 0; i < programCount; ++i) {
        if (a == (i + 1)) continue;
        // if all() doesn't return 'true', we have an error.
        if (!all(a != (i + 1))) {
            ++errs;
        }
    }

    RET[programIndex] = errs;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
