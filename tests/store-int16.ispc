export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int x[16] = 0xffffffff;
    unsigned int val = aFOO[programIndex];
    store_to_int16(x, 5, val);
    unsigned int v = load_from_int16(x, 6);
    RET[programIndex] = v;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2+programIndex;
    RET[programCount-1] = 0xffff;
}
