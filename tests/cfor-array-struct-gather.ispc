struct Foo {
    uniform float x[programCount+1];
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Foo foo;
    uniform int i;
    #pragma ignore warning
    cfor (i = 0; i < programCount+1; ++i)
        foo.x[i] = i;

    if ((int)a & 1) {
        #pragma ignore warning(perf)
        RET[programIndex] = foo.x[a-1]; return;
    }
    else {
        #pragma ignore warning(perf)
        RET[programIndex] = foo.x[a]; return;
    }
}


export void result(uniform float RET[]) {
    RET[programIndex] = ((programIndex+1) / 2) * 2;
}
