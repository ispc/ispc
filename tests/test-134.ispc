
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
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


export void result(uniform float RET[]) {
  for (int i = 0; i < programCount; i += 4)
  {
    RET[i+0] = 1;
    RET[i+1] = 3;
    RET[i+2] = 3;
    RET[i+3] = 29;
  }
}
