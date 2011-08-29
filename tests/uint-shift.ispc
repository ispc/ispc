
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    unsigned int x = 0xffffffff;
    RET[programIndex] = x >> 16; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0xffff;
}
