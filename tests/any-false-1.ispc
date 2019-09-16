export void f_f(uniform float RET[], uniform float aFOO[]) {

    int errs = 0;
    int a = aFOO[programIndex];
    if (programIndex > 2) {
        // if any() doesn't return 'false', we have an error.
        if (any(a <= 3)) {
            ++errs;
        }
    }


    RET[programIndex] = errs;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
