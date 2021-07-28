#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    int a = RET[programIndex] = aFOO[programIndex];

    int res = 0;
    if (a & 1) {
        foreach_active (i) {
            res = a;
            foreach_unique (c in a) {
                if (c % 3 == 1)
                    continue;
                res *= 2;
            }
        }
    }
    RET[programIndex] = res;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programIndex & 1) ? 0 :
                        ((programIndex % 3 == 0) ? (programIndex + 1) : 2 * (programIndex + 1));
}
