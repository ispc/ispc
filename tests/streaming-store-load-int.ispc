#include "../test_static.isph"
void var_store(uniform int a[], varying int x) {
    streaming_store(a, x);
}
uniform int var_load(uniform int a[], varying int x) {
    uniform int ret = 0;
    varying int loadval = streaming_load(a);
    if(loadval != x) ret=1;
    return ret;
}

void var_unsigned_store(uniform unsigned int a[], varying unsigned int x) {
    streaming_store(a, x);
}

uniform int var_unsigned_load(uniform unsigned int a[], varying unsigned int x) {
    uniform int ret = 0;
    varying unsigned int loadval = streaming_load(a);
    if(loadval != x) ret=1;
    return ret;
}

void uni_store(uniform int a[], uniform int x) {
    streaming_store(a, x);
}

uniform int uni_load(uniform int a[], uniform int x) {
    uniform int ret = 0;
    uniform int loadval = streaming_load_uniform(a);
    if(loadval != x) ret=1;
    return ret;
}

void uni_unsigned_store(uniform unsigned int a[], uniform unsigned int x) {
    streaming_store(a, x);
}

uniform int uni_unsigned_load(uniform unsigned int a[],uniform unsigned int x) {
    uniform int ret = 0;
    uniform unsigned int loadval = streaming_load_uniform(a);
    if(loadval != x) ret=1;
    return ret;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0;

    uniform int a_1[programCount];
    int x_1 = 1;
    var_store(a_1,x_1);
    if(var_load(a_1,x_1) != 0)RET[programIndex] = 1;

    uniform unsigned int a_2[programCount];
    unsigned int x_2 = 2;
    var_unsigned_store(a_2,x_2);
    if(var_unsigned_load(a_2,x_2) != 0)RET[programIndex] = 1;

    uniform int a_3[programCount];
    uniform int x_3 = 3;
    uni_store(a_3,x_3);
    if(uni_load(a_3,x_3) != 0)RET[programIndex] = 1;

    uniform unsigned int a_4[programCount];
    uniform unsigned int x_4 = 4;
    uni_unsigned_store(a_4,x_4);
    if(uni_unsigned_load(a_4,x_4) != 0)RET[programIndex] = 1;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
