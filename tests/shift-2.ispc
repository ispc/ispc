#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex]; 
    uniform int delta = b - 6; // -1
    int rot = shift(a, delta);
    RET[programIndex] = rot;
}

task void result(uniform float RET[]) {
    varying int val = programIndex;
    if (val < 0) val = 0;	 
    RET[programIndex] = val;	 
}
