export void f_v(uniform float RET[]) { if (true) { RET[programIndex] = (5%3) * (4./8.) + 2.14159; } else RET[programIndex] = 1.; }

export void result(uniform float RET[]) {
    RET[programIndex] = 3.141590;
}
