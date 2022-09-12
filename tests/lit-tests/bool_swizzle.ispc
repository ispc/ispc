// This test checks that compiler is not crashing on bool swizzle.
// Currently fails (https://github.com/ispc/ispc/issues/2367).

// RUN: %{ispc} %s --target=host --nostdlib
// XFAIL: *
export void test(uniform float out[]) {
    uniform bool<3> a = {1,0,1};
    uniform bool<3> b = a.zxy;
    out[programIndex] = b.x;
}
