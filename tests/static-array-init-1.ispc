#include "../test_static.isph"
static uniform float foo[4] = { 1, 2, 3, 4 };



task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    RET[programIndex] = foo[b-3]; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
