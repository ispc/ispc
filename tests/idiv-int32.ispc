#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
  uniform int errorCount = 0;

  uniform RNGState state;
  seed_rng(&state, 1234);

  for (uniform int64 i = 0; i < 64k; ++i) {
    int32 num = random(&state);
    if (num < 0)
      continue;
    for (uniform int32 div = 2; div < 256; ++div) {
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
