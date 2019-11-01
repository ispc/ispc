float bar(float x);
export void f_v(uniform float RET[]) { RET[programIndex] = bar(3.); }
float bar(float x) { return 2. * x; }

export void result(uniform float RET[]) {
    RET[programIndex] = 6.000000;
}
