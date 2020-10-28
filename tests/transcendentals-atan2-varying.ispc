// rule: skip on cpu=TGLLP
export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
  varying double arg = aFOO[programIndex];
  varying double arg2 = aFOO[programIndex] + b;
  varying double ref = atan2((float)arg, (float)arg2);
  varying double res = atan2(arg, arg2);
  varying double delta = abs((res - ref) / ref);
  RET[programIndex] = delta < 1d-6 ? 0.0 : 1.0;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0.0;
}
