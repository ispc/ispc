// Operators with templates specializations using implicit arguments are not supported yet.
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap  2>&1 | FileCheck %s -check-prefix=CHECK_ERROR

struct FMatrix {
    double M[16];
};

template <typename T> noinline FMatrix operator*(const FMatrix &A, const T &B) {
    return A;
}

// CHECK_ERROR: Template arguments deduction is not yet supported in template function specialization.
template <>
noinline FMatrix operator*(const FMatrix &A, const float &B) {
    return A;
}
