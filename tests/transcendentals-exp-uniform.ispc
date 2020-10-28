// rule: skip on cpu=TGLLP
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  for (uniform int i = 0; i != programCount; ++i) {
    uniform double ref = exp((float)aFOO[i]);
    uniform double delta = abs((exp(aFOO[i]) - ref) / ref);
    RET[i] = delta < 1d-6 ? 0.0 : 1.0;
  }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
