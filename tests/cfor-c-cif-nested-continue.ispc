static float float4(uniform float a, uniform float b, uniform float c, 
                    uniform float d) {
    float ret = 0;
    #pragma ignore warning
    cfor (uniform int i = 0; i < programCount; i += 4) {
        ret = insert(ret, i + 0, a);
        ret = insert(ret, i + 1, b);
        ret = insert(ret, i + 2, c);
        ret = insert(ret, i + 3, d);
    }
    return ret;
}




export void f_f(uniform float RET[], uniform float aFOO[]) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    int i, j;
    float r = 0;
    cfor (i = 0; i < a; ++i) {
        cif (i != 0) {
            cif (a != 2)
                continue;
            r = 10;
        }
        ++r;
    }
    RET[programIndex] = r;
}

export void result(uniform float RET[]) { 
    uniform float ret[4] = { 1,11,1,1 };
    #pragma ignore warning(perf)
    RET[programIndex] = ret[programIndex & 0x3];
}
