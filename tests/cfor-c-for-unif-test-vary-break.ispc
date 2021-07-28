#include "../test_static.isph"
static float float4(uniform float a, uniform float b, uniform float c, 
                    uniform float d) {
    float ret = 0;
    #pragma ignore warning
    cfor (uniform int i = 0; i < programCount; i += 4) {
        ret = insert(ret, i + 0, a);
        ret = insert(ret, i + 1, b);
        ret = insert(ret, i + 2, c);
        ret = insert(ret, i + 3, d);
    }
    return ret;
}




task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    uniform int x;
    float aa = a;
    cfor (x = 0; x < 99999; ++x) {
        if (x == a) break;
        ++aa;
    }
    RET[programIndex] = aa;
}

task void result(uniform float RET[]) {
    uniform float ret[4] = { 2,4,6,8 };
    #pragma ignore warning(perf)
    RET[programIndex] = ret[programIndex & 0x3];
}
