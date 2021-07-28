#include "../test_static.isph"
task void f_v(uniform float RET[]) {
int i;
cfor (i = 0; i < 10; ++i) {
  if (i == 5) break;
}
RET[programIndex] = (float)i; }

task void result(uniform float RET[]) {
    RET[programIndex] = 5.000000;
}
