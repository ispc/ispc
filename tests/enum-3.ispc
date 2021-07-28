#include "../test_static.isph"
enum Foo {
    ZERO,
    ONE,
    TWO,
    TEN = 10,
    ELEVEN
};


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    RET[programIndex] = ELEVEN; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 11;
}
