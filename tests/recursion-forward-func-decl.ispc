#include "../test_static.isph"
float f(float);

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex];
    #pragma ignore warning(perf)
    RET[programIndex] = f(a % 16);
}

float f(float x) {
    if (x == 0)  return x;
    else         return x + f(x-1);
}

task void result(uniform float RET[]) {
    uniform float fib[16] = { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 0 };
    RET[programIndex] = fib[programIndex % 16];
}
