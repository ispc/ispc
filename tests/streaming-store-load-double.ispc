// rule: skip on cpu=TGLLP

void var_store(uniform double a[], varying float x) {
    streaming_store(a, x);
}

uniform int var_load(uniform double a[], varying float x) {
    uniform int ret = 0;
    varying double loadval = streaming_load(a);
    if(loadval != x) ret=1;
    return ret;
}

void uni_store(uniform double a[],uniform double x) {
    streaming_store(a, x);
}

uniform int uni_load(uniform double a[],uniform double x) {
    uniform int ret = 0;
    uniform double loadval = streaming_load_uniform(a);
    if(loadval != x) ret=1;
    return ret;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0;

    uniform double a_1[programCount];
    double x_1 = 1;
    var_store(a_1,x_1);
    if(var_load(a_1,x_1) != 0)RET[programIndex] = 1;

    uniform double a_3[programCount];
    uniform double x_3 = 3;
    uni_store(a_3,x_3);
    if(uni_load(a_3,x_3) != 0)RET[programIndex] = 1;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
