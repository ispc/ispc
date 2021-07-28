#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    unsigned int x = 0xffff0f0f;
    #pragma ignore warning(perf)
    RET[programIndex] = ~x;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0xf0f0;
}
