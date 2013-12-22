
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    varying unsigned int8 a = 255, b = aFOO[programIndex]; // max unsigned int8
    RET[programIndex] = saturating_add(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = (varying unsigned int8) 255;
}
