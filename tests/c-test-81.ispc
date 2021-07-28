#include "../test_static.isph"
task void f_v(uniform float RET[]) {
int i=0, j=0;
while (i++ < 10) {
  ++j;
  if (i >= 5) break;
  ++j;}
RET[programIndex] = (float)j; }

task void result(uniform float RET[]) {
    RET[programIndex] = 9.000000;
}
