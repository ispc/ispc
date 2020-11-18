// rule: skip on arch=genx32
// rule: skip on arch=genx64
// test fails on gen and affects execution of other tests
export void f_f(uniform float RET[], uniform float aFOO[]) {
  uniform int errorCount = 0;

  // randomly sample int32s...
  uniform RNGState state;
  seed_rng(&state, 1234);
  for (uniform int i = 0; i < 64k; ++i) {
    unsigned int32 num = random(&state);
    for (uniform unsigned int32 div = 2; div < 256; ++div) {
      #pragma ignore warning(perf)
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        #pragma ignore warning(perf)
#ifndef ISPC_TARGET_GENX
        print("ui32 error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
#endif
        if (errorCount > 32) break;
      }
    }
  }

  RET[programIndex] = errorCount;
}

export void result(uniform float RET[]) {
  RET[programIndex] = 0;
}

