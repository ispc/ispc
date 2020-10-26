uniform unsigned int32 s = 0;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    int32 b = 0;
    if (programIndex < 32 && programIndex & 1)
        b = atomic_or_global(&s, (1ul << programIndex));
    RET[programIndex] = popcnt(reduce_max((int32)b));
}

export void result(uniform float RET[]) {
    RET[programIndex] = programCount == 1 ? 0 : ((min(32, programCount)/2) - 1);
}
