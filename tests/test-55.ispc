
export uniform int width() { return programCount; }


export void f_v(uniform float RET[]) { RET[programIndex] = 1.; } 
export void f_v(uniform float RET[]) { RET[programIndex] = 2.; }

export void result(uniform float RET[]) {
    RET[programIndex] = 1.000000;
}
