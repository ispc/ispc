
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform unsigned int8 a = 255, b = 255; // max unsigned int8
    RET[programIndex] = paddus(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 255;
}
