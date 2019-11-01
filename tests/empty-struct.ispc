struct Foo { };

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform Foo f;
    RET[programIndex] = sizeof(f); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
