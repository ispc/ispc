#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
  uniform int errorCount1 = 0;
  uniform int errorCount2 = 0;

  // uniform computation under varying condition should not be executed, as
  // none of the lanes compare as "false".
  if (aFOO[programIndex] < -1) {
    errorCount1++;
  }

  if (aFOO[programIndex] < -2) {
    ++errorCount1;
  }

  if (aFOO[programIndex] < -3) {
    errorCount2--;
  }

  if (aFOO[programIndex] < -4) {
    --errorCount2;
  }

  RET[programIndex] = errorCount1 - errorCount2;
}

task void result(uniform float RET[]) {
  RET[programIndex] = 0;
}

