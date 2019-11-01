export void f_v(uniform float RET[]) {
    float sum = 0;
    int errors = 0;

    // We should match up except for the denorms (0->1023) and the inf/nan
    // ranges.
    for (uniform int i = 1024; i < 32768-1024; ++i) {
        unsigned int16 h = i;
        float f = half_to_float_fast(i);
        h = float_to_half_fast(f);

        if (i != h)
            ++errors;
    }

    for (uniform int i = 32768+1024; i < 65536-1024; ++i) {
        unsigned int16 h = i;
        float f = half_to_float_fast(i);
        h = float_to_half_fast(f);

        if (i != h)
            ++errors;
    }
    RET[programIndex] = errors;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
