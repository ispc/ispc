#include "test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    // Use % 14 to limit input to 2^13 max, ensuring ldexp(a, 2) <= 32768 which
    // stays within float16's finite range (65504). Using % 28 would allow 2^14
    // and 2^15 inputs, causing ldexp results to overflow.
    float16 a = 1ul << ((programIndex/4) % 14);
    if (programIndex & 1)
        a = -a;
    RET[programIndex] = ldexp(a, 2);
}

task void result(uniform float RET[]) {
    int pi = (programIndex/4) % 14;
    #pragma ignore warning(perf)
    RET[programIndex] = (1ul << (pi + 2));
    if (programIndex & 1)
        RET[programIndex] = -RET[programIndex];
}
