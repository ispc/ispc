export void f_v(uniform float RET[]) {
    int i=0;
    float f=0;
    bool b=0;
    RET[programIndex] = b ? f : (i+1);
}


export void result(uniform float RET[]) {
    RET[programIndex] = 1.000000;
}
