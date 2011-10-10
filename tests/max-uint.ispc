
export uniform int width() { return programCount; }

export void f_f(uniform float result[], uniform float aa[]) {
    unsigned int i = (unsigned int)aa[programIndex];
    result[programIndex] = max((unsigned int)100, i);
}

export void result(uniform float r[]) { r[programIndex] = 100; }

