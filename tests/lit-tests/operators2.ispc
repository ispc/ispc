// RUN: %{ispc} --target=host --nowrap --emit-llvm-text -o - %s | FileCheck %s

struct Complex {
    float real;
    float imag;
};

// Arithmetic operators
noinline Complex operator+(Complex a, Complex b) {
    Complex result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

noinline Complex operator-(Complex a, Complex b) {
    Complex result;
    result.real = a.real - b.real;
    result.imag = a.imag - b.imag;
    return result;
}

// Comparison operators
noinline bool operator==(Complex a, Complex b) {
    return (a.real == b.real && a.imag == b.imag);
}

noinline bool operator!=(Complex a, Complex b) {
    return (a.real != b.real || a.imag != b.imag);
}

// Logical operators
noinline bool operator&&(Complex a, Complex b) {
    return (a.real != 0.0f || a.imag != 0.0f) && 
           (b.real != 0.0f || b.imag != 0.0f);
}

noinline bool operator||(Complex a, Complex b) {
    return (a.real != 0.0f || a.imag != 0.0f) || 
           (b.real != 0.0f || b.imag != 0.0f);
}

// CHECK-LABEL: @test_binary_ops
unmasked int test_binary_ops(float* uniform out) {
    Complex c1, c2;
    c1.real = 1.0f; c1.imag = 2.0f;
    c2.real = 3.0f; c2.imag = 4.0f;
    
    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator+___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex sum = c1 + c2;
    out[programIndex] += sum.real + sum.imag;
    // CHECK: call %v{{[0-9]*}}_varying_Complex @operator-___s_5B_vyComplex_5D_s_5B_vyComplex_5D_
    Complex diff = c1 - c2;
    out[programIndex] += diff.real + diff.imag;

    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator==___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    bool equal = (c1 == c2);
    
    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator!=___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    bool notEqual = (c1 != c2);
    
    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator&&___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    bool logicalAnd = (c1 && c2);
    
    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator||___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    bool logicalOr = (c1 || c2);

    return equal || notEqual || logicalAnd || logicalOr;
}