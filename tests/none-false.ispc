#include "../test_static.isph"
task void f_v(uniform float RET[]) {

     int errs = 0;
    if (programIndex < 3) {
        // if none() doesn't return 'false', we have an error.
        if (none(programIndex >= 2)) {
             ++errs;
        }
    }


    RET[programIndex] = errs;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
