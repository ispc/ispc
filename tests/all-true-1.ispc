export void f_f(uniform float RET[], uniform float aFOO[]) {

    int errs = 0;
    int a = aFOO[programIndex];
    if (programIndex < 2) {
        // if all() doesn't return 'true', we have an error.
        if (!all(a <= 2)) {
            ++errs;
        }
    }


    RET[programIndex] = errs;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
