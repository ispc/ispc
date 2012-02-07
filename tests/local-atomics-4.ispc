
export uniform int width() { return programCount; }

uniform int32 s = 0;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = atomic_or_local(&s, (1<<programIndex));
    RET[programIndex] = s;
}

export void result(uniform float RET[]) {
    RET[programIndex] = (1<<programCount)-1;
}
