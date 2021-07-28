#include "../test_static.isph"
struct Foo { float f; };

void f(Foo foo[], float a) {
    #pragma ignore warning(perf)
    ++foo[a].f;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    Foo foo[programCount+5];
    for (uniform int i = 0; i < programCount+5; ++i)
        foo[i].f = a;
    f(foo, a);
    #pragma ignore warning(perf)
    RET[programIndex] = foo[a].f;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 2+programIndex;
}
