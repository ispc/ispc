
export uniform int width() { return programCount; }



export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float v = aFOO[programIndex & 0x3];
    uniform float m;
    unsigned int iv = (unsigned int)v;
/*CO    if (iv & 1)*/
        m = reduce_add(iv);
    RET[programIndex] = m;
}

export void result(uniform float RET[]) { RET[programIndex] = (programCount == 1) ? 1 : (10 * programCount/4); }
