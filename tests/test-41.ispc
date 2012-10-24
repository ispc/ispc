
export uniform int width() { return programCount; }


export void f_v(uniform float RET[]) { 
    float x = 1.;  RET[programIndex] = (x == 1.) ? (x + 2.14159) : 0.; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 3.141590;
}
