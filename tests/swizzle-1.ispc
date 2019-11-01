export void f_v(uniform float RET[]) {
    float<3> a = {1,2,3};
    float<3> b = a.zxy;

    RET[programIndex] = b.x;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
