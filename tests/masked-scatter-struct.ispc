struct Foo { float x; float y; };

export void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    float a = aa[programIndex];
    uniform Foo foo[programCount+1];
    for (uniform int i = 0; i < programCount+1; ++i) {
        foo[i].x = i;
        foo[i].y = -1234 + i;
    }
    #pragma ignore warning(perf)
    varying Foo fv = foo[a];
    fv.x += 1000;
//CO    print("fv.x = %\n", fv.x);
//CO    print("foo[a] = %\n", foo[a].x);
    #pragma ignore warning(perf)
    foo[a] = fv;
//CO    print("after assign foo[a] = %\n", foo[a].x);

    #pragma ignore warning(perf)
    ret[programIndex] = foo[programIndex].x;
//CO    print("% - %\n", programIndex, ret[programIndex]);
}

export void result(uniform float ret[]) {
    ret[programIndex] = 1000+programIndex;
    ret[0] = 0;
}
