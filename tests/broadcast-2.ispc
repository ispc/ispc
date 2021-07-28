#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    int16 a = aFOO[programIndex]; 
    int16 b = (programCount == 1) ? 3 : broadcast(a, 2);
    RET[programIndex] = b;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
