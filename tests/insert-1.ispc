export void f_f(uniform float RET[], uniform float aFOO[]) {
    int64 a;
    for (uniform int i = 0; i < programCount; ++i)
        a = insert(a, i, (int64)i);
    RET[programIndex] = a; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
