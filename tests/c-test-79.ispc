#include "../test_static.isph"
task void f_v(uniform float RET[]) {
int i, j=0;
for (i = 0; i < 10; ++i) {
  if (i >= 5) continue;
  ++j;}
RET[programIndex] = (float)j; }

task void result(uniform float RET[]) {
    RET[programIndex] = 5.000000;
}
