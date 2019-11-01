export void f_v(uniform float RET[]) {
    float<3> a = {1,10,100};
    float<6> b = a.zxyyxz;

    RET[programIndex] = b.x + 2*b.y + 3*b.z - 3*b[3] - 2*b[4] - b[5];
}


export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
