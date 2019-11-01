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

export void f_v(uniform float RET[4]) {
    float a = float4((1.252511382102966),(8.731211662292480),(8.605137825012207),(2.428840160369873));
    float b = float4((-0.307321906089783),(0.000000000000000),(-9.626051902770996),(-3.708563089370728));
    float ref = float4((0.933145880699158),(1.000000000000000),(0.000000001004552),(0.037215489894152));
    RET[programIndex] = ok(pow(a, b), ref) ? 1. : 0.;
}
export void result(uniform float RET[]) { RET[programIndex] = 1.; }
