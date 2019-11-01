typedef int<3> int3;

export void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    float a = aa[programIndex];
    uniform int3 array[programCount+1];
    for (uniform int i = 0; i < programCount + 6 - b; ++i) {
        for (uniform int j = 0; j < 3; ++j)
            array[i][j] = i+100*j;
    }

    #pragma ignore warning(perf)
    varying int3 vv = array[a];
    ++vv.y;
    #pragma ignore warning(perf)
    array[a] = vv;
    #pragma ignore warning(perf)
    ret[programIndex] = array[programIndex].y;
}

export void result(uniform float ret[]) {
    ret[programIndex] = 101+programIndex;
    ret[0] = 100;
}
