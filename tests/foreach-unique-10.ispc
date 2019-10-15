export void f_f(uniform float RET[], uniform float aFOO[]) {
    int a = RET[programIndex] = aFOO[programIndex];

    int res = 0;
    if (a & 1) {
        foreach_active (i) {
            res = a;
            foreach_unique (c in a) {
                res *= 2;
            }
        }
    }
    RET[programIndex] = res;
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programIndex & 1) ? 0 : 2 * (programIndex + 1);
}
