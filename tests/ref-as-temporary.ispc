int func(const int &a) { return a+1; }
int bar() { return 0; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = func(bar());
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
