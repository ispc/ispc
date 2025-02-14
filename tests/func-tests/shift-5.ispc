#include "test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uint8 a = aFOO[programIndex]; 
    uint8 rot = shift(a, 1);
    RET[programIndex] = rot;
}

task void result(uniform float RET[]) {
    varying int val = 2 + programIndex;
    if (val > programCount) val = 0;	 
    RET[programIndex] = val;	 
}   
