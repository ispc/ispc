#include "../test_static.isph"
float f(int i) { return i + 1.; }
float f(float v) { return 2 * v; }
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = f(a) + f(10l);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 13 + 2 * programIndex;
}
