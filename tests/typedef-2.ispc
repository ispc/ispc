#include "../test_static.isph"
typedef float FLOAT;

struct Foo {
    FLOAT x[16];
};

typedef Foo FOO;

struct Bar {
    FOO foo[16];
};

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    Bar bar;
    for (uniform int i = 0; i < 16; ++i)
        for (uniform int j = 0; j < 16; ++j)
            bar.foo[i].x[j] = b;
    #pragma ignore warning(perf)
    RET[programIndex] = bar.foo[min(15.f, a-1)].x[min(15.f, a-1)];
}

task void result(uniform float RET[]) { RET[programIndex] = 5; }
