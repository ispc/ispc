float foo(float &a) { return 1; }
float foo(const float &a) { return 2; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float x = 0;
    RET[programIndex] = foo(x); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
