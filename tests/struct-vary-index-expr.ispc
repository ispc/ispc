struct Foo { float f; };

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    Foo foo[programCount+1];
    uniform int ind[programCount+1];
    for (uniform int i = 0; i < programCount+1; ++i) {
        foo[i].f = a;
        ind[i] = i+1;
    }
    #pragma ignore warning(perf) 
    ++foo[a].f;
    #pragma ignore warning(perf)
    RET[programIndex] = foo[ind[programIndex]].f;
}

export void result(uniform float RET[]) { RET[programIndex] = 2+programIndex; }
