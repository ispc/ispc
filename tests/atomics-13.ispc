
export uniform int width() { return programCount; }

uniform unsigned int32 s = 0;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = 0;
    if (programIndex & 1)
        b = atomic_or_global(&s, (1 << programIndex));
    RET[programIndex] = popcnt(reduce_max((int32)b));
}

export void result(uniform float RET[]) {
    RET[programIndex] = programCount == 1 ? 0 : ((programCount/2) - 1);
}
