export void f_v(uniform float RET[]) { RET[programIndex] = (2 < 3) ? 42. : -1.; }

export void result(uniform float RET[]) {
    RET[programIndex] = 42.000000;
}
