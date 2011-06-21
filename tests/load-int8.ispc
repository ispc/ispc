export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int x[4] = { 0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d };
    unsigned int v = load_from_int8(x, 0);
    RET[programIndex] = v;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
