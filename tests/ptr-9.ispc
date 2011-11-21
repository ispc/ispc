
export uniform int width() { return programCount; }

void inc(int * uniform v) {
    ++*v;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    int a = aFOO[programIndex];
    int * uniform b = &a;
    void * uniform vp = b;
    int * uniform c = (int * uniform)vp;
    RET[programIndex] = *c;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
