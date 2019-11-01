export void f_v(uniform float RET[]) {
    float<3> a = {1,2,3};
    float<3> b = a.rgb;

    RET[programIndex] = b.x + 2*b.y + 3*b.z;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 14;
}
