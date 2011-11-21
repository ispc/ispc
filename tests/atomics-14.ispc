
export uniform int width() { return programCount; }

uniform unsigned int64 s = 0xffffffffff000000;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = 0;
    if (programIndex & 1)
        b = atomic_or_global(&s, (1 << programIndex));
    RET[programIndex] = (s>>20);
}

export void result(uniform float RET[]) {
    uniform int sum = 0;
    for (uniform int i = 0; i < programCount; ++i)
        if (i & 1)
            sum += (1 << i);
    RET[programIndex] = ((unsigned int64)(0xffffffffff000000 | sum)) >> 20;
}
