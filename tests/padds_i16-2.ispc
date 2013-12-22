
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int16 a = -32768; // min signed int16
    RET[programIndex] = saturating_add(a, -b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = (uniform int16) -32768;
}
