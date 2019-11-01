struct Foo {
    int a;
    uniform float b;
};

void update(varying float<3> * uniform vp) {
    vp->x = 0;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float<3> v = { 1, 2, 3 };
    update(&v);
    RET[programIndex] = v.x;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
