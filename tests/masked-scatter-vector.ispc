
export uniform int width() { return programCount; }

typedef int<3> int3;

export void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    float a = aa[programIndex];
    uniform int3 array[32];
    for (uniform int i = 0; i < 6*b + 2; ++i) {
        for (uniform int j = 0; j < 3; ++j)
            array[i][j] = i+100*j;
    }

    varying int3 vv = array[a];
    ++vv.y;
    array[a] = vv;
    ret[programIndex] = array[programIndex].y;
}

export void result(uniform float ret[]) {
    ret[programIndex] = 101+programIndex;
    ret[0] = 100;
}
