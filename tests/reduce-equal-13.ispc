export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    int a = aFOO[programIndex&1];
    RET[programIndex] = 1;
    uniform bool re = false;
    uniform int val = 16;
    if (programIndex & 1) {
        re = reduce_equal(a, &val);
    }
    RET[programIndex] = ((int)re << 8) + val;
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programCount == 1) ? 16 : 258;
}
