export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    int i, j;
    float r = 0;
    for (i = 0; i < a; ++i) {
        cif (i != 0) {
            cif (a != 2)
                continue;
            r = 10;
        }
        ++r;
    }
    RET[programIndex] = r;
}

export void result(uniform float RET[]) { RET[programIndex] = 1; RET[1] = 11; }
