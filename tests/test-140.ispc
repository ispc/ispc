#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&3]; 
    RET[programIndex] = log(a); 
}

task void result(uniform float RET[]) {
  for (int i = 0; i < programCount; i += 4)
  {
    #pragma ignore warning
    RET[i+0] = 0x0.0p+0;
    #pragma ignore warning
    RET[i+1] = 0x1.62e43p-1;
    #pragma ignore warning
    RET[i+2] = 0x1.193ea8p+0;
    #pragma ignore warning
    RET[i+3] = 0x1.62e43p+0;
  }
}
