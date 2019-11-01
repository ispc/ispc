export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    int i, j;
    float r = 0;
    for (i = 0; i < a+1; ++i) {
        if (i == 1)
            continue;
        for (j = 0; j < a; ++j) {
            if (a == 2)
                continue;
        }
        ++r;
    }
    RET[programIndex] = r;
}

export void result(uniform float RET[]) { RET[programIndex] = 1+programIndex; }
