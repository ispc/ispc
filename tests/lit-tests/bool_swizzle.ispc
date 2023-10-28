// RUN: %{ispc} %s --target=host --nostdlib

export void test(uniform float out[]) {
    uniform bool<3> a = {1,0,1};
    uniform bool<3> b = a.zxy;
    out[programIndex] = b.x;
}
