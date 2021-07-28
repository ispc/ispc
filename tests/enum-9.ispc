#include "../test_static.isph"
enum {
    ZERO,
    ONE,
    TWO
};


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    RET[programIndex] = ONE; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
