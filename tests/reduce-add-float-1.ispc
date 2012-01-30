
export uniform int width() { return programCount; }



export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float v = aFOO[programIndex];
    uniform float m;
    int iv = (int)v;
    if (iv & 1)
        m = reduce_add((float)iv);
    RET[programIndex] = m;
}

export void result(uniform float RET[]) { 
    uniform int x = -1234;
    if (programCount == 1) x = 1;
    else if (programCount == 4) x = 4;
    else if (programCount == 8) x = 16;
    else if (programCount == 16) x = 64;
    RET[programIndex] = x;
}

