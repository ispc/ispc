// rule: skip on arch=x86
// rule: skip on arch=x86-64
export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float16 a = aFOO[programIndex];
    float16 rot = rotate(a, -1);
    RET[programIndex] = rot;
}

export void result(uniform float RET[]) { RET[programIndex] = 1 + (programIndex + programCount - 1) % programCount; }
