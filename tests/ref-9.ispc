int &foo(int &a, int &b) {
    return b;
}

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex];
    int three = 3;
    int &val = foo(a, three);
    RET[programIndex] = val;
}

export void result(uniform float RET[]) { 
    RET[programIndex] = 3;
}
