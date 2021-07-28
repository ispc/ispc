#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    #pragma ignore warning
    cif (-1 < 0) RET[programIndex] = 10.; 
    else RET[programIndex] = 0.; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10.000000;
}
