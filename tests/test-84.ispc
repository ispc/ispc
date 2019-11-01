float f(float v) { return 2 * v; }
float f(int i) { return i + 1.; }
export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = f(a) + f(10l);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 13 + 2 * programIndex;
}
