#include "../test_static.isph"
task void f_v(uniform float RET[]) {

    int errs = 0;
    if (programIndex < 2) {
        // if all() doesn't return 'true', we have an error.
        if (!all(programIndex < 2)) {
            ++errs;
        }
    }


    RET[programIndex] = errs;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
