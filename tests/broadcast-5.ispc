// rule: skip on arch=x86
// rule: skip on arch=x86-64
export void f_f(uniform float RET[], uniform float aFOO[]) {
    float16 a = aFOO[programIndex];
    float16 b = (programCount == 1) ? 4 : broadcast(a, 2);
    RET[programIndex] = b;
}

export void result(uniform float RET[]) { RET[programIndex] = 3; }
