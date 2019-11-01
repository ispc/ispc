export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform float<4> x = {1,2,3,4}, y = {b-5,b-5,b-5,b-5};

    RET[programIndex] = 0;

    int index = aFOO[programIndex];
    index = min(index, 3);

    #pragma ignore warning(perf)
    RET[programIndex] = x[index];


/*CO    return x << 1;*/
/*CO    return c[0] ? 1 : 0;*/
/*CO    x = b;*/
/*CO    y = b;*/
/*CO    return x+y;*/
}

export void result(uniform float RET[]) {
    RET[programIndex] = 4;
    RET[0] = 2;
    RET[1] = 3;
    RET[2] = 4;
}
