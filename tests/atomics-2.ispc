
export uniform int width() { return programCount; }

uniform int64 s = 0;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = atomic_add_global(s, 1);
    RET[programIndex] = reduce_add(b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = reduce_add(programIndex);
}
