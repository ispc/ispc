float foo(uniform float<4> &vec) {
    #pragma ignore warning(perf)
    return vec[programIndex & 3];
}

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform float<4> vec = { b, -1, 2*b, -b };
    RET[programIndex] = foo(vec); 
}

export void result(uniform float RET[]) {
    uniform float a[4] = { 5, -1, 10, -5 };
    #pragma ignore warning(perf)
    RET[programIndex] = a[programIndex & 3];
}
