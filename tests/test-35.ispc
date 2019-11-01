export void f_v(uniform float RET[]) { if (1. != 1) RET[programIndex] = -1.; else RET[programIndex] = 10.; }

export void result(uniform float RET[]) {
    RET[programIndex] = 10.000000;
}
