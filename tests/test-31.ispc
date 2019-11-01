export void f_v(uniform float RET[]) { if (true || 0.) RET[programIndex] = 10.; else RET[programIndex] = -1.; }

export void result(uniform float RET[]) {
    RET[programIndex] = 10.000000;
}
