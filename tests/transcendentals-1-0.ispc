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
    float v = float4((-3.141592741012573),(0.000000000000000),(3.141592741012573),(6.283185482025146));
    float ref = float4((-1.000000000000000),(1.000000000000000),(-1.000000000000000),(1.000000000000000));
    RET[programIndex] = ok(cos(v), ref) ? 1. : 0.;
}
export void result(uniform float RET[]) { RET[programIndex] = 1.; }
