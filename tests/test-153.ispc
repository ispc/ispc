#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = aFOO[programIndex]; 
    RET[programIndex] = ceil(-a+1e-5); 
}

task void result(uniform float RET[]) {
    RET[programIndex] = - (double) programIndex;
}
