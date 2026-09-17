// Regression test for passing a bool short-vector reference to a conversion function.
// RUN: %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll

int<3> convert(const bool<3> v) {
    int<3> result;

    result.x = v.x ? 1 : 0;
    result.y = v.y ? 1 : 0;
    result.z = v.z ? 1 : 0;

    return result;
}

void f(const bool<3>& v) {
    int<3> i = convert(v);
}
