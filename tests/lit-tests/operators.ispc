// RUN: %{ispc} --target=host --nowrap -O0 --emit-llvm-text -o - %s | FileCheck %s

struct Complex {
    float real;
    float imag;
};

// Binary arithmetic operators
noinline Complex operator*(Complex a, Complex b) {
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

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

noinline Complex operator/(Complex a, Complex b) {
    float denom = b.real * b.real + b.imag * b.imag;
    Complex result;
    result.real = (a.real * b.real + a.imag * b.imag) / denom;
    result.imag = (a.imag * b.real - a.real * b.imag) / denom;
    return result;
}

noinline Complex operator%(Complex a, Complex b) {
    // Simple example - just perform modulo on real parts
    Complex result;
    result.real = a.real - (int)(a.real / b.real) * b.real;
    result.imag = a.imag;
    return result;
}

// Bitshift operators (treating real part as an integer)
noinline Complex operator<<(Complex a, Complex b) {
    Complex result;
    result.real = (int)a.real << (int)b.real;
    result.imag = a.imag;
    return result;
}

noinline Complex operator>>(Complex a, Complex b) {
    Complex result;
    result.real = (int)a.real >> (int)b.real;
    result.imag = a.imag;
    return result;
}

// Comparison operators
noinline bool operator==(Complex a, Complex b) {
    return (a.real == b.real && a.imag == b.imag);
}

noinline bool operator!=(Complex a, Complex b) {
    return (a.real != b.real || a.imag != b.imag);
}

noinline bool operator>(Complex a, Complex b) {
    // Compare magnitude
    float mag_a = a.real * a.real + a.imag * a.imag;
    float mag_b = b.real * b.real + b.imag * b.imag;
    return mag_a > mag_b;
}

noinline bool operator<(Complex a, Complex b) {
    float mag_a = a.real * a.real + a.imag * a.imag;
    float mag_b = b.real * b.real + b.imag * b.imag;
    return mag_a < mag_b;
}

noinline bool operator>=(Complex a, Complex b) {
    float mag_a = a.real * a.real + a.imag * a.imag;
    float mag_b = b.real * b.real + b.imag * b.imag;
    return mag_a >= mag_b;
}

noinline bool operator<=(Complex a, Complex b) {
    float mag_a = a.real * a.real + a.imag * a.imag;
    float mag_b = b.real * b.real + b.imag * b.imag;
    return mag_a <= mag_b;
}

// Bitwise operators (treating real parts as integers)
noinline Complex operator&(Complex a, Complex b) {
    Complex result;
    result.real = (int)a.real & (int)b.real;
    result.imag = (int)a.imag & (int)b.imag;
    return result;
}

noinline Complex operator^(Complex a, Complex b) {
    Complex result;
    result.real = (int)a.real ^ (int)b.real;
    result.imag = (int)a.imag ^ (int)b.imag;
    return result;
}

noinline Complex operator|(Complex a, Complex b) {
    Complex result;
    result.real = (int)a.real | (int)b.real;
    result.imag = (int)a.imag | (int)b.imag;
    return result;
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

// Unary operators
noinline Complex operator++(Complex &a) {
    a.real += 1.0f;
    return a;
}

noinline Complex operator--(Complex &a) {
    a.real -= 1.0f;
    return a;
}

noinline Complex operator++(Complex &a, int) {
    Complex temp = a;
    a.real += 1.0f;
    return temp;
}

noinline Complex operator--(Complex &a, int) {
    Complex temp = a;
    a.real -= 1.0f;
    return temp;
}

noinline Complex operator~(Complex a) {
    Complex result;
    result.real = ~(int)a.real;
    result.imag = ~(int)a.imag;
    return result;
}

noinline bool operator!(Complex a) {
    return (a.real == 0.0f && a.imag == 0.0f);
}

// Assignment operators
noinline Complex operator=(Complex &a, Complex b) {
    a.real = b.real;
    a.imag = b.imag;
    return a;
}

noinline Complex operator*=(Complex &a, Complex b) {
    float temp_real = a.real * b.real - a.imag * b.imag;
    a.imag = a.real * b.imag + a.imag * b.real;
    a.real = temp_real;
    return a;
}

noinline Complex operator/=(Complex &a, Complex b) {
    float denom = b.real * b.real + b.imag * b.imag;
    float temp_real = (a.real * b.real + a.imag * b.imag) / denom;
    a.imag = (a.imag * b.real - a.real * b.imag) / denom;
    a.real = temp_real;
    return a;
}

noinline Complex operator%=(Complex &a, Complex b) {
    a.real = a.real - (int)(a.real / b.real) * b.real;
    return a;
}

noinline Complex operator+=(Complex &a, Complex b) {
    a.real += b.real;
    a.imag += b.imag;
    return a;
}

noinline Complex operator-=(Complex &a, Complex b) {
    a.real -= b.real;
    a.imag -= b.imag;
    return a;
}

noinline Complex operator<<=(Complex &a, Complex b) {
    a.real = (int)a.real << (int)b.real;
    return a;
}

noinline Complex operator>>=(Complex &a, Complex b) {
    a.real = (int)a.real >> (int)b.real;
    return a;
}

noinline Complex operator&=(Complex &a, Complex b) {
    a.real = (int)a.real & (int)b.real;
    a.imag = (int)a.imag & (int)b.imag;
    return a;
}

noinline Complex operator|=(Complex &a, Complex b) {
    a.real = (int)a.real | (int)b.real;
    a.imag = (int)a.imag | (int)b.imag;
    return a;
}

noinline Complex operator^=(Complex &a, Complex b) {
    a.real = (int)a.real ^ (int)b.real;
    a.imag = (int)a.imag ^ (int)b.imag;
    return a;
}

// CHECK-LABEL: @test_binary_ops
unmasked void test_binary_ops(uniform float* uniform out) {
    Complex c1, c2;
    c1.real = 5.0f; c1.imag = 2.0f;
    c2.real = 3.0f; c2.imag = 4.0f;
    float res;
    // Arithmetic operators
    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator*___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex mul = c1 * c2;
    res = mul.real + mul.imag;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator+___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex sum = c1 + c2;
    res += sum.real + sum.imag;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @operator-___s_5B_vyComplex_5D_s_5B_vyComplex_5D_
    Complex diff = c1 - c2;
    res += diff.real + diff.imag;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator/___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex div = c1 / c2;
    res += div.real + div.imag;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator%___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex mod = c1 % c2;
    res += mod.real + mod.imag;

    // Bitshift operators
    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator<<___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex lshift = c1 << c2;
    res += lshift.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator>>___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex rshift = c1 >> c2;
    res += rshift.real;

    // Comparison operators
    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator==___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    bool equal = (c1 == c2);
    res += equal ? 1.0f : 0.0f;

    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator!=___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    bool notEqual = (c1 != c2);
    res += notEqual ? 1.0f : 0.0f;

    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator>___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    bool greater = (c1 > c2);
    res += greater ? 1.0f : 0.0f;

    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator<___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    bool less = (c1 < c2);
    res += less ? 1.0f : 0.0f;

    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator>=___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    bool greaterOrEqual = (c1 >= c2);
    res += greaterOrEqual ? 1.0f : 0.0f;

    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator<=___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    bool lessOrEqual = (c1 <= c2);
    res += lessOrEqual ? 1.0f : 0.0f;

    // Bitwise operators
    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator&___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex bitAnd = c1 & c2;
    res += bitAnd.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator^___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex bitXor = c1 ^ c2;
    res += bitXor.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator|___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex bitOr = c1 | c2;
    res += bitOr.real;

    // Logical operators
    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator&&___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    bool logicalAnd = (c1 && c2);
    res += logicalAnd ? 1.0f : 0.0f;

    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator||___s_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    bool logicalOr = (c1 || c2);
    res += logicalOr ? 1.0f : 0.0f;
    out[programIndex] = res;
}

// CHECK-LABEL: @test_unary_ops
unmasked void test_unary_ops(uniform float* uniform out) {
    Complex c;
    c.real = 5.0f; c.imag = 2.0f;
    float res;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator++___REFs_5B_vyComplex_5D_"
    Complex preinc = ++c;
    res += preinc.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @operator--___REFs_5B_vyComplex_5D_
    Complex predec = --c;
    res += predec.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator++___REFs_5B_vyComplex_5D_vyi"
    Complex postinc = c++;
    res += postinc.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @operator--___REFs_5B_vyComplex_5D_vyi
    Complex postdec = c--;
    res += postdec.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator~___s_5B_vyComplex_5D_"
    Complex bitnot = ~c;
    res += bitnot.real;

    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator!___s_5B_vyComplex_5D_"
    bool lognot = !c;
    res += lognot ? 1.0f : 0.0f;
    out[programIndex] = res;
}

// CHECK-LABEL: @test_assignment_ops
unmasked void test_assignment_ops(float* uniform out) {
    Complex c1, c2;
    c1.real = 5.0f; c1.imag = 2.0f;
    c2.real = 3.0f; c2.imag = 4.0f;

    float res;
    // Assignment operators
    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator=___REFs_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex assign;
    assign = c1;
    res += assign.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator*=___REFs_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex c3 = c1;
    c3 *= c2;
    res += c3.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator/=___REFs_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex c4 = c1;
    c4 /= c2;
    res += c4.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator%=___REFs_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex c5 = c1;
    c5 %= c2;
    res += c5.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator+=___REFs_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex c6 = c1;
    c6 += c2;
    res += c6.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator-=___REFs_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex c7 = c1;
    c7 -= c2;
    res += c7.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator<<=___REFs_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex c8 = c1;
    c8 <<= c2;
    res += c8.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator>>=___REFs_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex c9 = c1;
    c9 >>= c2;
    res += c9.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator&=___REFs_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex c10 = c1;
    c10 &= c2;
    res += c10.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator|=___REFs_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex c11 = c1;
    c11 |= c2;
    res += c11.real;

    // CHECK: call %v{{[0-9]*}}_varying_Complex @"operator^=___REFs_5B_vyComplex_5D_s_5B_vyComplex_5D_"
    Complex c12 = c1;
    c12 ^= c2;
    res += c12.real;
    out[programIndex] = res;
}
