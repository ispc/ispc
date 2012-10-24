
export uniform int width() { return programCount; }

struct Foo { float x; float y; };

export void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    float a = aa[programIndex];
    uniform Foo foo[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        foo[i].x = i;
        foo[i].y = -1234 + i;
    }
    varying Foo fv = foo[a];
    fv.x += 1000;
//CO    print("fv.x = %\n", fv.x);
//CO    print("foo[a] = %\n", foo[a].x);
    foo[a] = fv;
//CO    print("after assign foo[a] = %\n", foo[a].x);

    ret[programIndex] = foo[programIndex].x;
//CO    print("% - %\n", programIndex, ret[programIndex]);
}

export void result(uniform float ret[]) {
    ret[programIndex] = 1000+programIndex;
    ret[0] = 0;
}
