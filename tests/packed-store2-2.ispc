export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    uniform int pack[2+programCount];
    uniform int number;
    for (uniform int i = 0; i < 2+programCount; ++i)
        pack[i] = 0;
    if ((int)a & 1)
        number = packed_store_active2(&pack[2], a);
    pack[2+number] = 0;
    RET[programIndex] = pack[programIndex]; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
    uniform int val = 1;
    for (uniform int i = 2; i < 2+programCount/2; ++i, val += 2)
        RET[i] = val;
}
