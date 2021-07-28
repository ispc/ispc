#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    // Neither of these should hit!
    assert(a > 0);
    assert(b == 5.);
    RET[programIndex] = 2*a; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
