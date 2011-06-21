export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int x[8] = { 0x00020001, 0x00040003, 0x00060005, 0x00080007,
                         0x000a0009, 0x000c000b, 0x000e000d, 0x0010000f };
    unsigned int v = load_from_int16(x, 0);
    RET[programIndex] = v;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
