#define N 64
export void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 uniform buf[N];
    memset(buf, 0, programCount*sizeof(uniform int32));
    memset(buf+1, 0x7f, (programCount-2)*sizeof(uniform int32));

    int v = buf[programIndex];
    RET[programIndex] = (v == 0x7f7f7f7f);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
    RET[0] = 0;
    RET[programCount-1] = 0;
}
