export void f_v(uniform float RET[]) {
    int errors = 0;

    foreach (i = 0 ... 65535) {
        unsigned int16 h = i;
        float f = half_to_float(i);
        h = float_to_half(f);

        int mismatches = (!isnan(f) && i != h);
        if (any(mismatches != 0))
#ifndef ISPC_TARGET_GENX
            print("mismatch: orig int16 % -> float % -> half %\n", i, f, h);
#endif
        errors += reduce_add(mismatches);
    }

    RET[programIndex] = errors;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
