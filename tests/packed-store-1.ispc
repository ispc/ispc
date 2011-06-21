
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    uniform int pack[2+programCount] = 0;
    packed_store_active(pack, 2, a);
    RET[programIndex] = pack[programIndex]; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex-1;
    RET[0] = RET[1] = 0;
}
