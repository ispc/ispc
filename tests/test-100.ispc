
export uniform int width() { return programCount; }


static float array[6];
task void x(uniform int i, float f) {
    array[i] = f;
}
export void f_fu(uniform float RET[], uniform float fFOO[], uniform float fu) { 
    float f = fFOO[programIndex];
    launch < x(fu, f) >;
    sync;
    RET[programIndex] = array[5];
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
