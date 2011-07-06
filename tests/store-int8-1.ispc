export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform unsigned int x[8];
    for (uniform int i = 0; i < 8; ++i)
        x[i] = 0xffffffff;
    unsigned int val = aFOO[programIndex];
    store_to_uint8(x, 2, val);
    unsigned int v = load_from_uint8(x, 1);
    RET[programIndex] = v;
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
    RET[0] = (unsigned int)0xff;
}
