
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float bb) {
    unsigned int16 a = aFOO[programIndex], b = bb;
    RET[programIndex] = ((unsigned int16)4000*a)+b; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = (((4000*(programIndex+1))&0xffff)+5)&0xffff;
}
