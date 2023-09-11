// Basic check of template operators (issue #2634)
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// CHECK: define %v{{[0-9]*}}_varying_FMatrix @"operator*___s_5B_vyFMatrixF_5D____REFs_5B__c_vyFMatrix_5D_REFs_5B__c_vyFMatrixF_5D_"({{.*}} %A, {{.*}} %B, <{{[0-9]*}} x {{.*}}> %__mask)
// CHECK: define weak_odr %v{{[0-9]*}}_varying_FMatrix @"operator*___s_5B_vyFMatrixI_5D____REFs_5B__c_vyFMatrix_5D_REFs_5B__c_vyFMatrixI_5D_"({{.*}} %A, {{.*}} %B, <{{[0-9]*}} x {{.*}}> %__mask)

// CHECK-LABEL: define <{{[0-9]*}} x double> @DoMath___
// CHECK: call %v{{[0-9]*}}_varying_FMatrix @"operator*___{{.*}}"({{.*}} %A, {{.*}} %B, <{{[0-9]*}} x {{.*}}> %__mask)

// CHECK-LABEL: define linkonce_odr %v{{[0-9]*}}_varying_FMatrix @"operator*___{{.*}}"({{.*}} %A, {{.*}} %B, <{{[0-9]*}} x {{.*}}> %__mask)


struct FMatrix {
    double M[16];
};

struct FMatrixF {
    float M[16];
};

struct FMatrixI {
    int M[16];
};

template <typename T> noinline FMatrix operator*(const FMatrix &A, const T &B) {
    FMatrix Result;

    for (uniform unsigned int m = 0; m < 4; m++) {
        varying double Sum;
        for (uniform unsigned int k = 0; k < 4; k++) {
            Sum = 0;
            for (uniform unsigned int n = 0; n < 4; n++) {
                Sum += A.M[m * 4 + n] * B.M[n * 4 + k];
            }

            Result.M[m * 4 + k] = Sum;
        }
    }

    return A;
}

template <>
noinline FMatrix operator*<FMatrixF>(const FMatrix &A, const FMatrixF &B) {
    return A;
}

template FMatrix noinline operator*<FMatrixI>(const FMatrix &A, const FMatrixI &B);

double DoMath() {
    FMatrix A;
    FMatrix B;
    FMatrix result = A * B;
    return result.M[0];
}
