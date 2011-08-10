
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex&1];
    RET[programIndex] = 1;
    uniform bool re;
    uniform int val;
    if (programIndex & 1) {
        re = reduce_equal(a, val);
    }
    RET[programIndex] = ((int)re << 8) + val;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 258;
}
