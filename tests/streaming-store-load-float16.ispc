#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
void var_store(uniform float16 a[], varying float x) {
    streaming_store(a, x);
}

uniform int var_load(uniform float16 a[], varying float x) {
    uniform int ret = 0;
    varying float16 loadval = streaming_load(a);
    if(loadval != x) ret=1;
    return ret;
}

void uni_store(uniform float16 a[], uniform float16 x) {
    streaming_store(a, x);
}

uniform int uni_load(uniform float16 a[], uniform float16 x) {
    uniform int ret = 0;
    uniform float16 loadval = streaming_load_uniform(a);
    if(loadval != x) ret=1;
    return ret;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0;

    uniform float16 a_1[programCount];
    float16 x_1 = 1.0f16;
    var_store(a_1,x_1);
    if(var_load(a_1,x_1) != 0)RET[programIndex] = 1;

    uniform float16 a_3[programCount];
    uniform float16 x_3 = 3.0f16;
    uni_store(a_3,x_3);
    if(uni_load(a_3,x_3) != 0)RET[programIndex] = 1;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
