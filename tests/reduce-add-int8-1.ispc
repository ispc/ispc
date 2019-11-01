export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float v = aFOO[programIndex];
    uniform float m;
    int8 iv = (int)v;
    if (iv & 1)
        m = reduce_add(iv);
    RET[programIndex] = m;
}

export void result(uniform float RET[]) { 
    uniform int x = 0;
    for (uniform int i = 1; i <= programCount; i += 2)
        x += i;
    RET[programIndex] = x;
}

