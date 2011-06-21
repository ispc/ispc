
export uniform int width() { return programCount; }

struct Foo { float x; float y; };

export void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    float a = aa[programIndex];
    Foo foo[32];
    for (uniform int i = 0; i < 32; ++i)
        foo[i].x = i;
    varying Foo fv = foo[a];
    fv.x += 1000;
//CO    print("fv.x = %\n", fv.x);
//CO    print("foo[a] = %\n", foo[a].x);
    foo[a] = fv;
//CO    print("after assign foo[a] = %\n", foo[a].x);

    ret[programIndex] = foo[a].x;
//CO    print("% - %\n", programIndex, ret[programIndex]);
}

export void result(uniform float ret[]) {
    ret[programIndex] = 1001+programIndex;
}
