// rule: skip on arch=x86
// rule: skip on arch=x86-64
export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float16 a = aFOO[programIndex];
    int reverse = programCount - 1 - programIndex + (int)b - 5;
    float16 shuf = shuffle(a, reverse);
    RET[programIndex] = shuf;
}

export void result(uniform float RET[]) { RET[programIndex] = programCount - programIndex; }
