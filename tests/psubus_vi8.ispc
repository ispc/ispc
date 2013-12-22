
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    varying unsigned int8 a = 0, b = aFOO[programIndex]; // min unsigned int8
    RET[programIndex] = saturating_sub(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = (varying unsigned int8) 0;
}
