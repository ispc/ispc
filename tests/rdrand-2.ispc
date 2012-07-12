
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
#if !defined(ISPC_TARGET_AVX11) && !defined(ISPC_TARGET_AVX2)
    RET[programIndex] = 1;
#else

    float r = -1;
    while (!rdrand(&r))
        ;
    RET[programIndex] = (r >= 0 && r < 1);

#endif
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
