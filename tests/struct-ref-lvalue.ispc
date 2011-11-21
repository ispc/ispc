
export uniform int width() { return programCount; }



struct Foo { float f; };

void f(uniform Foo foo[], float a) {
    ++foo[a].f;
}

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Foo foo[17];
    for (uniform int i = 0; i < 17; ++i)
        foo[i].f = a;
    f(foo, a);
    RET[programIndex] = foo[a].f;
}

export void result(uniform float RET[]) { 
    RET[programIndex] = 2+programIndex;
}
