uniform int32 s = 0xff;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    int32 bits = 0xfff0;
    float b = atomic_xor_global(&s, bits);
    RET[programIndex] = s;
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programCount & 1) ? 0xff0f : 0xff;
}
