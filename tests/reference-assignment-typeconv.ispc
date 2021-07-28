#include "../test_static.isph"
void foo(float &x, int y) {
    x = y;
}

task void f_fu(uniform float ret[], uniform float a[], uniform float b) {
    float aa = a[programIndex];
    int bb = (int)b;
    foo(aa, bb);
    ret[programIndex] = aa;
}

task void result(uniform float r[]) {
    r[programIndex] = 5;
}
