
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    uniform unsigned int pack[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        pack[i] = 0;
    packed_store_active(pack, (unsigned int)a);
    RET[programIndex] = pack[programIndex]; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
