
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int8 a = 0; // min unsigned int8
    RET[programIndex] = saturating_sub(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = (uniform unsigned int8) 0;
}
