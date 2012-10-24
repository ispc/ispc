
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float bb) {
    unsigned int8 a = aFOO[programIndex], b = bb;
    RET[programIndex] = ((unsigned int8)100*a)+b; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = (((100*(programIndex+1))&0xff)+5)&0xff;
}
