
export uniform int width() { return programCount; }

uniform unsigned int32 s[programCount];

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = 0;
    float delta = 1;
    if (programIndex < 2)
        atomic_add_global(&s[programIndex], delta);
    RET[programIndex] = s[programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = RET[1] = 1;
}
