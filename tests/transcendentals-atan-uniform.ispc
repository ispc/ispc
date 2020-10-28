// rule: skip on cpu=TGLLP
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform double arg = aFOO[i];
    uniform double ref = atan((float)arg);
    uniform double res = atan(arg);
    uniform double delta = abs((res - ref) / ref);
    RET[i] = delta < 1d-7 ? 0.0 : 1.0;
  }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
