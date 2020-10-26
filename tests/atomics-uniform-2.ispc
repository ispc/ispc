uniform unsigned int32 s = 0b1010;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    uniform unsigned int32 b = atomic_or_global(&s, 1);
    RET[programIndex] = s;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0b1011;
}
