#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    RET[programIndex] = 0;
    if (true) return; 
    else RET[programIndex] = 1234.; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0.000000;
}
