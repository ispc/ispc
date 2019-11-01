extern int foo;
int foo = 1;

export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = foo; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
