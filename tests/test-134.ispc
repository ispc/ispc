#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    float i, j;
    for (i = 0; i < b; ++i) {
        cif (a == 1.) break;
        for (j = 0; j < b; ++j) {
            if (a == 3.) break;
            ++a;
        }
    }
    RET[programIndex] = a;
}


task void result(uniform float RET[]) {
  for (int i = 0; i < programCount; i += 4)
  {
    #pragma ignore warning
    RET[i+0] = 1;
    #pragma ignore warning
    RET[i+1] = 3;
    #pragma ignore warning
    RET[i+2] = 3;
    #pragma ignore warning
    RET[i+3] = 29;
  }
}
