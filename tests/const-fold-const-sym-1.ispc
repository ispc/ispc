export void f_v(uniform float RET[]) {
    const uniform float x = 1.;
    RET[programIndex] = x+x;
}


export void result(uniform float RET[]) { RET[programIndex] = 2; }
