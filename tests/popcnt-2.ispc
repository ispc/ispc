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
    for (uniform int i = 0; i < programCount; i += 4) {
        RET[i] = 16;
        RET[i+1] = 8;
        RET[i+2] = 1;
        RET[i+3] = 0;
    }
}
