int foo(uniform float * uniform base, uniform int uOfs, varying int vOfs) {
    #pragma ignore warning(perf)
    return (base+uOfs)[vOfs];
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform ptr = &aFOO[0];
    int val = foo(ptr, programCount, programIndex);
    RET[programIndex] = val;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programCount+programIndex;
}
