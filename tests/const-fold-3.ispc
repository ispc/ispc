export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    uniform int x = (17 < 2) || (6 >= 5) && (20 >= 20);
    const static uniform int y = (17 < 2) || (6 >= 5) && (20 >= 20);
    RET[programIndex] = ((x!=0) == (y!=0)) ? 1. : 0.;
}

export void result(uniform float RET[]) { RET[programIndex] = 1.; }
