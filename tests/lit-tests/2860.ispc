// Check that the code below can be compiled successfully.
// RUN: %{ispc} --target=host --nowrap -o %t.o %s
// CHECK-NOT: Error:
template <typename TVec, typename T> TVec FastDiv(T Lhs, TVec Rhs) {
    TVec Result = {Lhs * rcp((T)Rhs.X), Lhs * rcp((T)Rhs.Y)};
    return Result;
}

struct FVec {
    float X;
    float Y;
};

void foo(float Lhs, FVec Rhs) { FastDiv<FVec, float>(Lhs, Rhs); }