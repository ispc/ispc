export void f_v(uniform float RET[]) { float x = 10; RET[programIndex] = --x; }

export void result(uniform float RET[]) {
    RET[programIndex] = 9.000000;
}
