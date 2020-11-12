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



bool ok(float x, float ref) { 
    bool r = (abs(x - ref) < 1e-4) || abs((x-ref)/ref) < 1e-4;
    if (any(r == false))
#ifndef ISPC_TARGET_GENX
        print("mismatch got %, expected %\n", x, ref);
#endif
    return r;
}

export void f_v(uniform float RET[]) {
    float v = float4((-9.424777984619141),(4.000000000000000),(10.000000000000000),(-10.000000000000000));
    float ref = float4((0.000000023849761),(-0.756802499294281),(-0.544021129608154),(0.544021129608154));
    RET[programIndex] = ok(sin(v), ref) ? 1. : 0.;
}
export void result(uniform float RET[]) { RET[programIndex] = 1.; }
