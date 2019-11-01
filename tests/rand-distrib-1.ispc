export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    RNGState state;
    seed_rng(&state, programIndex);
    int count[32];
    for (uniform int i = 0; i < 32; ++i)
        count[i] = (b == 5.) ? 0 : 1;
    uniform int iters = 10000;
    for (uniform int i = 0; i < iters; ++i) {
        unsigned int val = random(&state);
        for (uniform int j = 0; j < 32; ++j) {
            if (val & (1ul<<j))
                ++count[j];
        }
    }

    bool ok = true;
    for (uniform int i = 0; i < 32; ++i) 
        ok |= (count[i] > 0.495 * iters && count[i] < 0.505 * iters);
    RET[programIndex] = ok ? 1 : 0;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
