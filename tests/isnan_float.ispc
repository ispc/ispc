export void f_v(uniform float RET[]) {
    int errors = 0;
    for (uniform int i = -2; i <= 2; ++i) {
        float f = log((float)i);

        if ((i < 0) && (!isnan(f))) errors ++;
        if ((i >= 0) && isnan(f))   errors ++;
    }

    for (uniform int i = -2; i <= 2; ++i) {
        uniform float f = log((uniform float)i);
        
        if ((i < 0) && (!isnan(f))) errors ++;
        if ((i >= 0) && isnan(f))   errors ++;
    }

    RET[programIndex] = errors;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
