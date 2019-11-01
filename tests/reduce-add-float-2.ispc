export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float v = aFOO[programIndex];
    uniform float m;
    int iv = (int)v;
    m = reduce_add((float)iv);
    RET[programIndex] = m;
}

export void result(uniform float RET[]) { 
    uniform int x = 0;
    for (uniform int i = 1; i <= programCount; ++i)
        x += i;
    RET[programIndex] = x;
}

