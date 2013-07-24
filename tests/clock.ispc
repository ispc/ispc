
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
  unsigned uniform int64 a = clock();
  float x = pow(sqrt(aFOO[programIndex]), 5.5);
  unsigned uniform int64 b = clock();
  RET[programIndex] = (b - a) > 0 ? 1 : 0;
}

export void result(uniform float RET[]) {
  RET[programIndex] = 1;
}
