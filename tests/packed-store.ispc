
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    uniform int pack[programCount] = 0;
    packed_store_active(pack, 0, a);
    RET[programIndex] = pack[programIndex]; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
