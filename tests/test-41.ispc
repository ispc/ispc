#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    float x = 1.;  RET[programIndex] = (x == 1.) ? (x + 2.14159) : 0.; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 3.141590;
}
