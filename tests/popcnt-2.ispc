
export uniform int width() { return programCount; }

static int int4(uniform int a, uniform int b, uniform int c, 
                uniform int d) {
    int ret = 0;
    for (uniform int i = 0; i < programCount; i += 4) {
        ret = insert(ret, i + 0, a);
        ret = insert(ret, i + 1, b);
        ret = insert(ret, i + 2, c);
        ret = insert(ret, i + 3, d);
    }
    return ret;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    RET[programIndex] = popcnt(int4(0xf0f0f0f0, 0xff, 0x10, 0));
}

export void result(uniform float RET[]) { 
    RET[0] = RET[4] = RET[8] = RET[12] = 16;
    RET[1] = RET[5] = RET[9] = RET[13] = 8;
    RET[2] = RET[6] = RET[10] = RET[14] = 1;
    RET[3] = RET[7] = RET[11] = RET[15] = 0;
}
