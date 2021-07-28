#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {

    int errs = 0;
    int a = aFOO[programIndex];

    for (int i = 0; i < programCount; ++i) {
        bool check = false;
        if (a == (i + 1)) check = true;
        // if all() doesn't return 'false', we have an error.
        if (all(check)) {
            ++errs;
        }
    }

    RET[programIndex] = errs;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
