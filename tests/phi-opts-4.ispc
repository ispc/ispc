export void f_f(uniform float RET[], uniform float aFOO[]) {
    float sum = 0;
    // Obscfucated way to make delta all one, but make that unclear to the
    // compiler
    #pragma ignore warning(perf)
    int delta = aFOO[min(0., aFOO[programIndex])];

    // The optimization shouldn't apply for this, since delta isn't known
    // to be all equal
    for (int i = 0; i < 16; i += delta) {
        #pragma ignore warning(perf)
        sum += aFOO[i];
    }
    RET[programIndex] = extract(sum, 0);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 136;
}
