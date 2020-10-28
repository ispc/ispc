// Conversion between incompatible pointer types

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int8 * varying pa = (uniform int8 *)aFOO;
    RET[programIndex] = aFOO - pa;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 40;
}
