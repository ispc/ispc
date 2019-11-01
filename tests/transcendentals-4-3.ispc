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

export void f_v(uniform float RET[]) {
    float v = float4((14.300010190734863),(6.700009809265136),(21.200010762939453),(9.000010000000000));
    float ref = float4((2.660260200500488),(1.902109026908875),(3.054001569747925),(2.197225570678711));
    RET[programIndex] = ok(log(v), ref) ? 1. : 0.;
}
export void result(uniform float RET[4]) { RET[programIndex] = 1.; }
