float foo(float &) { return 1; }
float bar(uniform float []) { return 2; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float x = 0;
    RET[programIndex] = foo(x) + bar(aFOO);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
