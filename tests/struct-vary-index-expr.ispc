
export uniform int width() { return programCount; }



struct Foo { float f; };

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Foo foo[17];
    for (uniform int i = 0; i < 17; ++i)
        foo[i].f = a;
    ++foo[a].f;
    uniform int i[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    RET[programIndex] = foo[i[programIndex]].f;
}

export void result(uniform float RET[]) { RET[programIndex] = 2+programIndex; }
