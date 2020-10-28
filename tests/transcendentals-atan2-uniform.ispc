// rule: skip on cpu=TGLLP
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform double arg = aFOO[i];
    uniform double arg2 = aFOO[i] + b;
    uniform double ref = atan2((float)arg, (float)arg2);
    uniform double res = atan2(arg, arg2);
    uniform double delta = abs((res - ref) / ref);
    RET[i] = delta < 1d-6 ? 0.0 : 1.0;
  }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
