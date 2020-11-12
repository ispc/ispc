// rule: skip on arch=genx32
// rule: skip on arch=genx64
// test execution is too slow on gen currently
export void f_f(uniform float RET[], uniform float aFOO[]) {
  uniform int errorCount = 0;

  for (unsigned int16 num = 0; num < 0xffff; ++num) {
    for (uniform unsigned int16 div = 2; div < 256; ++div) {
      #pragma ignore warning(perf)
      if (__fast_idiv(num, div) != num/div) {
        ++errorCount;
        #pragma ignore warning(perf)
#ifndef ISPC_TARGET_GENX
        print("error %/% = %, got %\n", num, div, num/div, __fast_idiv(num,div));
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

