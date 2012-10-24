
export uniform int width() { return programCount; }


float f(int i) { return i + 1.; }
float f(float v) { return 2 * v; }
export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = f(a) + f(10); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 13 + 2 * programIndex;
}
