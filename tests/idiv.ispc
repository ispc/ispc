
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
  uniform int errorCount = 0;
 
  for (unsigned int8 num = 0; num < 255; ++num) {
    for (uniform unsigned int8 div = 2; div < 255; ++div) {
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        print("error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
      }
    }
  }

  for (int8 num = 0; num < 127; ++num) {
    for (uniform int8 div = 2; div < 127; ++div) {
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        print("error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
      }
    }
  }

  for (int16 num = 0; num < 32767; ++num) {
    for (uniform int16 div = 2; div < 256; ++div) {
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        print("error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
      }
    }
  }

  for (unsigned int16 num = 0; num < 0xffff; ++num) {
    for (uniform unsigned int16 div = 2; div < 256; ++div) {
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        print("error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
      }
    }
  }

  // randomly sample int32s...
  uniform RNGState state;
  seed_rng(&state, 1234);
  for (uniform int i = 0; i < 64k; ++i) {
    unsigned int32 num = random(&state);
    for (uniform unsigned int32 div = 2; div < 256; ++div) {
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        print("ui32 error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
      }
    }
  }

  for (uniform int64 i = 0; i < 64k; ++i) {
    int32 num = random(&state);
    if (num < 0)
      continue;
    for (uniform int32 div = 2; div < 256; ++div) {
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        print("si32 error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
      }
    }
  }

  RET[programIndex] = errorCount;
}

export void result(uniform float RET[]) {
  RET[programIndex] = 0;
}

