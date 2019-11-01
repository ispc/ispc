const uniform float x = 1.;

export void f_v(uniform float RET[]) {
    RET[programIndex] = x+x;
}


export void result(uniform float RET[]) { RET[programIndex] = 2; }
