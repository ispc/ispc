export void f_f(uniform float RET[], uniform float aFOO[]) {
    float sum1 = 0, sum2 = 0;
    foreach (x = 0 ... 10, i = 0 ... 6) {
        sum1 += aFOO[i];
    }

    for (uniform int x = 0; x < 10; ++x) {
        for (uniform int i = 0; i < 6; i += programCount) {
            int index = i + programIndex;
            if (index < 6)
                sum2 += aFOO[index];
        }
    }

    RET[programIndex] = sum1 - sum2;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
