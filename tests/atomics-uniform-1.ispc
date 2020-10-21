// rule: skip on arch=genx32
// rule: skip on arch=genx64
uniform unsigned int32 s = 10;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    uniform unsigned int32 b = atomic_add_global(&s, 1);
    RET[programIndex] = s;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 11;
}
