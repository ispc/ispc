
export uniform int width() { return programCount; }



struct Foo {
    uniform float x[17];
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Foo foo;
    uniform int i;
    cfor (i = 0; i < 17; ++i)
        foo.x[i] = i;

    if ((int)a & 1)
        { RET[programIndex] = foo.x[a-1]; return; }
    else 
        { RET[programIndex] = foo.x[a]; return; }
}


export void result(uniform float RET[]) {
    RET[programIndex] = ((programIndex+1) / 2) * 2;
}
