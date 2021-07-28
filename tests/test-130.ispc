#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    a = insert(a, (uniform int)b-5, 1234.);
    RET[programIndex] = extract(a, 0);
}


task void result(uniform float RET[]) {
    RET[programIndex]=1234;
}
