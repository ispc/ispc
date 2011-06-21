static float float4(uniform float a, uniform float b, uniform float c, 
                    uniform float d) {
    float ret = 0;
    for (uniform int i = 0; i < programCount; i += 4) {
        ret = insert(ret, i + 0, a);
        ret = insert(ret, i + 1, b);
        ret = insert(ret, i + 2, c);
        ret = insert(ret, i + 3, d);
    }
    return ret;
}

export uniform int width() { return programCount; }



export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform int ret = 0;

    float v = float4(1,1,0,0);
    bool b = (v == 1.);
    ret = __movmsk(((int)b));
    RET[programIndex] = ret;
}


// fixme for 16-wide...
export void result(uniform float RET[]) { 
    uniform int x = -1234;
    if (programCount == 4) x = 3;
    else if (programCount == 8) x = 0x33;
    else if (programCount == 16) x = 0x3333;
    RET[programIndex] = x;
}
