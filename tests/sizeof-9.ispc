#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
  RET[programIndex] = sizeof 1u;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 4;
}
