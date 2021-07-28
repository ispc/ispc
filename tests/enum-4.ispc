#include "../test_static.isph"
enum Foo {
    ZERO,
    ONE,
    TWO,
    TEN = 10,
    ELEVEN
};

enum Bar {
    FIFTY = 50,
    THOUSAND = 1000
};


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    Foo foo = (Foo)((a == 1.) ? ELEVEN : FIFTY);
    RET[programIndex] = (int)foo; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 50;
    RET[0] = 11;
}
