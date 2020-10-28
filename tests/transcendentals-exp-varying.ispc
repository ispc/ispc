// rule: skip on cpu=TGLLP
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  varying double ref = exp((float)aFOO[programIndex]);
  varying double res = exp(aFOO[programIndex]);
  varying double delta = abs((res - ref) / ref);
  RET[programIndex] = delta < 1d-6 ? 0.0 : 1.0;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
