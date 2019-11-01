export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform float udx[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        udx[i] = i+1;
    int x = -5 + programIndex + (int)b;
    RET[programIndex] = udx[x];
}

export void result(uniform float RET[4]) { RET[programIndex] = 1+programIndex; }
