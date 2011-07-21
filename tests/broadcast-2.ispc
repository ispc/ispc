
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    int16 a = aFOO[programIndex]; 
    int16 b = broadcast(a, 2);
    RET[programIndex] = b;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
