
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    varying unsigned int16 a = 0, b = aFOO[programIndex]; // min unsigned int16
    RET[programIndex] = saturating_sub(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = (varying unsigned int16) 0;
}
