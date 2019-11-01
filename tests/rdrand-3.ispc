export void f_f(uniform float RET[], uniform float aFOO[]) {
#ifndef ISPC_HAS_RAND
    RET[programIndex] = 1;
#else

    int lessHalf = 0, moreHalf = 0;
    for (uniform int i = 0; i < 1024*1024; ++i) {
        float r = -1;
        while (!rdrand(&r))
            ;
        if (r < 0.5) ++lessHalf;
        else ++moreHalf;
    }

    float r = (double)lessHalf / (double)(lessHalf + moreHalf);
    RET[programIndex] = (r >= .49 && r < .51);
#endif
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
