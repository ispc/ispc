export void f_f(uniform float RET[], uniform float aFOO[]) {
    static int a = 1234;
    {
        #pragma ignore warning
        static int a = 0;
        ++a;
        RET[programIndex] = a;
    }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
