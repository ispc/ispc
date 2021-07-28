#include "../test_static.isph"
static float float4(uniform float a, uniform float b, uniform float c, 
                    uniform float d) {
    float ret = 0;
    for (uniform int i = 0; i < programCount; i += 4) {
        ret = insert(ret, i + 0, a);
        ret = insert(ret, i + 1, b);
        ret = insert(ret, i + 2, c);
        ret = insert(ret, i + 3, d);
    }
    return ret;
}



bool ok(float x, float ref) { return (abs(x - ref) < 1e-6) || abs((x-ref)/ref) < 1e-5; }

task void f_v(uniform float RET[]) {
    float a = float4((6.008551597595215),(8.858951568603516),(3.376619100570679),(7.535532951354980));
    float b = float4((2.174313068389893),(-3.906857967376709),(-2.250460863113403),(2.072321891784668));
    float ref = float4((49.350250244140625),(0.000198935274966),(0.064665369689465),(65.714691162109375));
    RET[programIndex] = ok(pow(a, b), ref) ? 1. : 0.;
}
task void result(uniform float RET[]) { RET[programIndex] = 1.; }
