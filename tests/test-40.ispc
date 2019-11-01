export void f_v(uniform float RET[]) { RET[programIndex] = 1 + ((true == false) ? -1. : .14159) + 2.; }

export void result(uniform float RET[]) {
    RET[programIndex] = 3.141590;
}
