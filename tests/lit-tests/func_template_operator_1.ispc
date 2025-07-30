// Basic check of template operators (issue #2634)
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// Multiplication operator checks
// CHECK: define %v{{[0-9]*}}_varying_FMatrix @"operator*___s_5B_vyFMatrixF_5D____REFs_5B__c_vyFMatrix_5D_REFs_5B__c_vyFMatrixF_5D_"({{.*}} %A, {{.*}} %B, <{{[0-9]*}} x i{{[0-9]*}}> %__mask)

// Prefix increment operator checks
// CHECK: define %v{{[0-9]*}}_varying_FMatrixF @"operator++___s_5B_vyFMatrixF_5D____REFs_5B_vyFMatrixF_5D_"({{.*}} %A, <{{[0-9]*}} x i{{[0-9]*}}> %__mask)

// Postfix increment operator checks
// CHECK: define %v{{[0-9]*}}_varying_FMatrixF @"operator++___s_5B_vyFMatrixF_5D____REFs_5B_vyFMatrixF_5D_vyi"({{.*}} %A, <{{[0-9]*}} x i32> %__anon_parameter_1, <{{[0-9]*}} x i{{[0-9]*}}> %__mask)

// Assignment operator checks
// CHECK: define %v{{[0-9]*}}_varying_FMatrix @"operator=___s_5B_vyFMatrixF_5D____REFs_5B_vyFMatrix_5D_REFs_5B__c_vyFMatrixF_5D_"({{.*}} %A, {{.*}} %B, <{{[0-9]*}} x i{{[0-9]*}}> %__mask)

// CHECK-LABEL: define <{{[0-9]*}} x double> @DoMath___
// CHECK: call %v{{[0-9]*}}_varying_FMatrix @"operator*___{{.*}}"({{.*}} %A, {{.*}} %B, <{{[0-9]*}} x i{{[0-9]*}}> %__mask)

// CHECK-LABEL: define <{{[0-9]*}} x double> @TestIncrementOperators___
// CHECK: call %v{{[0-9]*}}_varying_FMatrixF @"operator++___s_5B_vyFMatrixF_5D____REFs_5B_vyFMatrixF_5D_"({{.*}} %A, <{{[0-9]*}} x i{{[0-9]*}}>
// CHECK: call %v{{[0-9]*}}_varying_FMatrixF @"operator++___s_5B_vyFMatrixF_5D____REFs_5B_vyFMatrixF_5D_vyi"({{.*}} %A, <{{[0-9]*}} x i32> {{.*}}, <{{[0-9]*}} x i{{[0-9]*}}>
// CHECK: call %v{{[0-9]*}}_varying_FMatrix @"operator=___s_5B_vyFMatrixF_5D____REFs_5B_vyFMatrix_5D_REFs_5B__c_vyFMatrixF_5D_"({{.*}} %D, {{.*}} %A, <{{[0-9]*}} x i{{[0-9]*}}>

struct FMatrix {
    double M[16];
};

struct FMatrixF {
    float M[16];
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

// Prefix increment operator (++) template
template <typename T> noinline T operator++(T &A) {
    // Simple implementation that increments all elements by 1
    for (uniform unsigned int i = 0; i < 16; i++) {
        A.M[i] += 1;
    }
    return A;
}

template <>
noinline FMatrixF operator++<FMatrixF>(FMatrixF &A) {
    for (uniform unsigned int i = 0; i < 16; i++) {
        A.M[i] += 1.0f;
    }
    return A;
}

// Postfix increment operator (i++) template
template <typename T> noinline T operator++(T &A, int) {
    T temp = A;
    ++A; // Reuse prefix increment
    return temp;
}

template <>
noinline FMatrixF operator++<FMatrixF>(FMatrixF &A, int) {
    FMatrixF temp = A;
    for (uniform unsigned int i = 0; i < 16; i++) {
        A.M[i] += 1.0f;
    }
    return temp;
}

// Assignment operator template
template <typename T> noinline FMatrix operator=(FMatrix &A, const T &B) {
    // Copy elements with type conversion if needed
    for (uniform unsigned int i = 0; i < 16; i++) {
        A.M[i] = B.M[i];
    }
    return A;
}

template <>
noinline FMatrix operator=<FMatrixF>(FMatrix &A, const FMatrixF &B) {
    for (uniform unsigned int i = 0; i < 16; i++) {
        A.M[i] = B.M[i];
    }
    return A;
}

double DoMath() {
    FMatrix A;
    FMatrix B;
    FMatrix result = A * B;
    return result.M[0];
}

double TestIncrementOperators() {
    FMatrixF A;
    for (uniform unsigned int i = 0; i < 16; i++) {
        A.M[i] = i;
    }

    // Test prefix increment
    FMatrixF B = ++A;

    // Test postfix increment
    FMatrixF C = A++;

    // Test assignment
    FMatrix D;
    D = A;

    return B.M[0] + C.M[0] + D.M[0];
}
