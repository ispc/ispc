void foo() { }
export void f_v(uniform float RET[]) { foo(); RET[programIndex] = 1.; }

export void result(uniform float RET[]) {
    RET[programIndex] = 1.000000;
}
