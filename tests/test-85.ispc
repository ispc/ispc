#include "../test_static.isph"
float f(float v) { return 2 * v; }
float f() { return 2.; }
float f(float a, int b) { return a + b; }
float f(int i) { return i + 1.; }
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = f(a) + f() + f(a, a) + f(10l);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 17 + 4 * programIndex;
}
