#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    int i=0, j=0;
    do {
        ++j;
        if (i >= 5) break;
        ++j;
        } while (++i < 10);
    RET[programIndex] = j;
}

task void result(uniform float RET[]) { RET[programIndex] = 11; }
