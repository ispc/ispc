
export uniform int width() { return programCount; }

uniform int32 s = 0xff;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    int32 bits = 0xfffffff0;
    float b = atomic_xor_global(s, bits);
    RET[programIndex] = s;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0xff;
}
