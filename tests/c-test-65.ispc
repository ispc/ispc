export void f_fu(uniform float RET[4], uniform float aFOO[4], uniform float b) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    float i, j;
    for (i = 0; i < b; ++i) {
        if (a == 1.) break;
        for (j = 0; j < b; ++j) {
            if (a == 3.) break;
            ++a;
        }
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 3;
    for (int i = 0; i < programCount; i += 4)
    {
      #pragma ignore warning
      RET[i+0] = 1;
      #pragma ignore warning
      RET[i+3] = 29;
    }
}
