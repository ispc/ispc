struct Foo {
    float x;
    float f;
    int i[3];
};
float bar(struct Foo f) { return f.f; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    varying Foo myFoo[3] = { { a, a, {a, a, a} }, 
                             { a, a, {a, a, a} }, 
                             { a, a, {a, a, a} } };
    RET[programIndex] = bar(myFoo[1]);
}


export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
