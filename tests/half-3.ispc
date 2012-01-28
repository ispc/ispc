
export uniform int width() { return programCount; }

export void f_v(uniform float RET[]) {
    int errors = 0;

    foreach (i = 0 ... 65535) {
        unsigned int16 h = i;
        float f = half_to_float(i);
        h = float_to_half(f);

        int mismatches = (f == f && i != h);
        errors += reduce_add(mismatches);
    }

    RET[programIndex] = errors;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
