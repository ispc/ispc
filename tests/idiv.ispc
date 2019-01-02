
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
  uniform int errorCount = 0;

  for (unsigned int8 num = 0; num < 255; ++num) {
    for (uniform unsigned int8 div = 2; div < 255; ++div) {
      #pragma ignore warning(perf)
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        #pragma ignore warning(perf)
        print("error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
        if (errorCount > 32) break;
      }
    }
  }

  for (int8 num = 0; num < 127; ++num) {
    for (uniform int8 div = 2; div < 127; ++div) {
      #pragma ignore warning(perf)
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        #pragma ignore warning(perf)
        print("error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
        if (errorCount > 64) break;
      }
    }
  }

  for (int16 num = 0; num < 32767; ++num) {
    for (uniform int16 div = 2; div < 256; ++div) {
      #pragma ignore warning(perf)
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        #pragma ignore warning(perf)
        print("error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
        if (errorCount > 96) break;
      }
    }
  }

  for (unsigned int16 num = 0; num < 0xffff; ++num) {
    for (uniform unsigned int16 div = 2; div < 256; ++div) {
      #pragma ignore warning(perf)
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        #pragma ignore warning(perf)
        print("error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
        if (errorCount > 128) break;
      }
    }
  }

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
        print("ui32 error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
        if (errorCount > 160) break;
      }
    }
  }

  for (uniform int64 i = 0; i < 64k; ++i) {
    int32 num = random(&state);
    if (num < 0)
      continue;
    for (uniform int32 div = 2; div < 256; ++div) {
      #pragma ignore warning(perf)
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        #pragma ignore warning(perf)
        print("si32 error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
        if (errorCount > 192) break;
      }
    }
  }

  RET[programIndex] = errorCount;
}

export void result(uniform float RET[]) {
  RET[programIndex] = 0;
}

