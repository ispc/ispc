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
    float a = float4((4.593603134155273),(9.463705062866211),(0.074282616376877),(0.242991551756859));
    float b = float4((3.049768924713135),(-2.639202594757080),(0.331999003887177),(1.839089989662170));
    float ref = float4((104.571922302246094),(0.002654463518411),(0.421828269958496),(0.074139028787613));
    RET[programIndex] = ok(pow(a, b), ref) ? 1. : 0.;
}
task void result(uniform float RET[]) { RET[programIndex] = 1.; }
