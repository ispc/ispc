#include "../test_static.isph"
task void f_v(uniform float RET[]) {
  int16 num = 1;
  uniform int16 div = 1;
  RET[programIndex] = __fast_idiv(num, div);
}

task void result(uniform float RET[]) {
  RET[programIndex] = 1;
}
