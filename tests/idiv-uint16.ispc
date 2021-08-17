#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
  uniform int errorCount = 0;

  for (unsigned int16 num = 0; num < 0xffff; ++num) {
    for (uniform unsigned int16 div = 2; div < 256; ++div) {
      #pragma ignore warning(perf)
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        if (errorCount > 32) break;
      }
    }
  }

  RET[programIndex] = errorCount;
}

task void result(uniform float RET[]) {
  RET[programIndex] = 0;
}
