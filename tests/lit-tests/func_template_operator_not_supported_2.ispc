// Operators are not supported with templates specializations yet.
// Need to support function templates with implicit template parameters first.
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap  2>&1 | FileCheck %s

// CHECK-NOT: Please file a bug report

// CHECK: Error: syntax error, unexpected identifier, expecting TOKEN_TEMPLATE_NAME
struct FMatrix {
    double M[16];
};

template <typename T> noinline FMatrix operator*(const FMatrix &A, const T &B) {
    return A;
}

template <>
noinline FMatrix operator*(const FMatrix &A, const int &B) {
    FMatrix Result;

    for (uniform unsigned int m = 0; m < 4; m++) {
        varying double Sum;
        for (uniform unsigned int k = 0; k < 4; k++) {
            Sum = 0;
            for (uniform unsigned int n = 0; n < 4; n++) {
                Sum += A.M[m * 4 + n] * B;
            }

            Result.M[m * 4 + k] = Sum;
        }
    }

    return Result;
}