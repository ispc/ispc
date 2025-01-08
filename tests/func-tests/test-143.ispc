#include "test_static.isph"
// rule: skip on target=generic.*
// rule: run on OS=!windows
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = round(a+.5); 
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programIndex / 2) * 2 + 2;
}
