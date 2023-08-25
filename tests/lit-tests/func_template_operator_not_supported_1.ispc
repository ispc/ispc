// Operators are not supported with explicit template instantiations yet.
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap  2>&1 | FileCheck %s

// CHECK-NOT: Please file a bug report

// CHECK: Error: syntax error, unexpected identifier, expecting TOKEN_TEMPLATE_NAME
struct FMatrix {
    double M[16];
};

template <typename T> noinline FMatrix operator*(const FMatrix &A, const T &B) {
    return A;
}

template noinline FMatrix operator*(const FMatrix &A, const int &B)