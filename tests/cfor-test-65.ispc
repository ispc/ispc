export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    float i, j;
    cfor (i = 0; i < b; ++i) {
        if (a == 1.) break;
        cfor (j = 0; j < b; ++j) {
            if (a == 3.) break;
            ++a;
        }
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
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
