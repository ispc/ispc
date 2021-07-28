#include "../test_static.isph"
task void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    uniform float foo[16];
    for (uniform int i = 0; i < 16; ++i)
        foo[i] = i;

    uniform int i = 0;
    foo[i++] += 1;
    ret[programIndex] = i;
}

task void result(uniform float ret[]) {
    ret[programIndex] = 1;
}
