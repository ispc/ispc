#include "../test_static.isph"
void foo(varying float * uniform x) {
    if ((*x) <= 2)
        ++(*x);
}

task void f_fu(uniform float ret[], uniform float a[], uniform float b) {
    float aa = a[programIndex];
    foo(&aa);
    ret[programIndex] = aa;
}

task void result(uniform float r[]) {
    r[programIndex] = 1+programIndex;
    r[0] = 2;
    r[1] = 3;
}
