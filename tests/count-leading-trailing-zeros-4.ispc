#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 i = (1ul << (programIndex % 28));
    RET[programIndex] = count_leading_zeros(i);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 31-(programIndex%28);
}
