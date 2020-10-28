// rule: skip on cpu=TGLLP
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform double arg = aFOO[i] / 16.0;
    uniform double ref = asin((float)arg);
    uniform double res = asin(arg);
    uniform double delta = abs((res - ref) / ref);
    RET[i] = delta < 1d-5 ? 0.0 : 1.0;
  }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
