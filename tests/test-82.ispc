#include "../test_static.isph"
void foo() { }
task void f_v(uniform float RET[]) { foo(); RET[programIndex] = 1.; }

task void result(uniform float RET[]) {
    RET[programIndex] = 1.000000;
}
