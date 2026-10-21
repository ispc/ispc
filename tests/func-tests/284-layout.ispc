#include "test_static.isph"
// Resolving a forward-declared struct through a pointer to it must not change
// the layout of another struct with the same name.
// See https://github.com/ispc/ispc/issues/284 and
// https://github.com/ispc/ispc/issues/3959

uniform bool compareFirst(void *uniform v) {
    struct L;
    uniform L *uniform p = (uniform L *uniform)v;
    struct L { int x; };
    return p != NULL;
}

uniform int64 compareSecond(void *uniform v) {
    struct L { int64 y; };
    uniform L *uniform p = (uniform L *uniform)v;
    return (p + 1)->y;
}

task void f_v(uniform float RET[]) {
    uniform int64 a[2] = {1, 2};
    uniform bool notNull = compareFirst(a);
    RET[programIndex] = notNull ? (float)compareSecond(a) : 0;
}

task void result(uniform float RET[]) { RET[programIndex] = 2; }
