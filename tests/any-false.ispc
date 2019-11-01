export void f_v(uniform float RET[]) {

    int errs = 0;
    if (programIndex > 2) {
        // if any() doesn't return 'false', we have an error.
        if (any(programIndex <= 2)) {
            ++errs;
        }
    }


    RET[programIndex] = errs;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
