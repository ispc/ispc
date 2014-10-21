
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex&3]; 
    RET[programIndex] = log(a); 
}

export void result(uniform float RET[]) {
  for (int i = 0; i < programCount; i += 4)
  {
    RET[i+0] = 0x0.0p+0;
    RET[i+1] = 0x1.62e43p-1;
    RET[i+2] = 0x1.193ea8p+0;
    RET[i+3] = 0x1.62e43p+0;
  }
}
