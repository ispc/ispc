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


Foo Func(Foo f) { return ++f; }
Bar Func(Bar b) { return FIFTY; }

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    RET[programIndex] = (int)Func(ONE);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 2;
}
