// Test for templates with type alias non-type parameters (size_t, ptrdiff_t, intptr_t, uintptr_t).
// RUN: %{ispc} %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// CHECK-LABEL: define {{.*}} @test_size_t
// CHECK: call {{.*}} @foo_size_t___Cuni10___
// CHECK: call {{.*}} @foo_size_t___Cuni42___

// CHECK-LABEL: define linkonce_odr {{.*}} @foo_size_t___Cuni10___
// CHECK: ret {{.*}} 10
// CHECK-LABEL: define linkonce_odr {{.*}} @foo_size_t___Cuni42___
// CHECK: ret {{.*}} 42

template<size_t N>
noinline size_t foo_size_t() {
    return N;
}

size_t test_size_t() {
    return foo_size_t<10>() + foo_size_t<42>();
}

// CHECK-LABEL: define {{.*}} @test_ptrdiff_t
// CHECK: call {{.*}} @foo_ptrdiff_t___Cuni20___
// CHECK: call {{.*}} @foo_ptrdiff_t___Cuni15___

// CHECK-LABEL: define linkonce_odr {{.*}} @foo_ptrdiff_t___Cuni20___
// CHECK: ret {{.*}} 20
// CHECK-LABEL: define linkonce_odr {{.*}} @foo_ptrdiff_t___Cuni15___
// CHECK: ret {{.*}} 15

template<ptrdiff_t N>
noinline ptrdiff_t foo_ptrdiff_t() {
    return N;
}

ptrdiff_t test_ptrdiff_t() {
    return foo_ptrdiff_t<20>() + foo_ptrdiff_t<15>();
}

// CHECK-LABEL: define {{.*}} @test_intptr_uintptr
// CHECK: call {{.*}} @foo_intptr___Cuni100___
// CHECK: call {{.*}} @foo_uintptr___Cuni200___

// CHECK-LABEL: define linkonce_odr {{.*}} @foo_intptr___Cuni100___
// CHECK: ret {{.*}} 100
// CHECK-LABEL: define linkonce_odr {{.*}} @foo_uintptr___Cuni200___
// CHECK: ret {{.*}} 200

template<intptr_t N>
noinline intptr_t foo_intptr() {
    return N;
}

template<uintptr_t N>
noinline uintptr_t foo_uintptr() {
    return N;
}

intptr_t test_intptr_uintptr() {
    return foo_intptr<100>() + (intptr_t)foo_uintptr<200>();
}

// Test: size_t template with specialization
// CHECK-LABEL: define {{.*}} @bar___Cuni0___
// CHECK: ret {{.*}} 99

// CHECK-LABEL: define {{.*}} @test_specialization
// CHECK: call {{.*}} @bar___Cuni1___

// CHECK-LABEL: define linkonce_odr {{.*}} @bar___Cuni1___
// CHECK: ret {{.*}} 1

template<size_t N>
noinline size_t bar() {
    return N;
}

template<>
noinline size_t bar<0>() {
    return 99;
}

size_t test_specialization() {
    return bar<0>() + bar<1>();
}

// Test: mix of type parameter, size_t, and enum
// CHECK-LABEL: define {{.*}} @test_mixed
// CHECK: call {{.*}} @mixed___vyiCuni5Cunenum_5B_Mode_5D_1___

// CHECK-LABEL: define linkonce_odr {{.*}} @mixed___vyiCuni5Cunenum_5B_Mode_5D_1___
// CHECK: ret {{.*}} {{i32 6|splat \(i32 6\)}}

enum Mode { OFF, ON };

template<typename T, size_t N, Mode M>
noinline T mixed() {
    return (T)(N + M);
}

int test_mixed() {
    return mixed<int, 5, ON>();
}

// Test: size_t used to create short vectors (T<N>)
// CHECK-LABEL: define {{.*}} @test_shortvec
// CHECK: call {{.*}} @shortvec_bar___vyiCuni3___

// CHECK-LABEL: define linkonce_odr {{.*}} @shortvec_bar___vyiCuni3___
// CHECK: ret

template<typename T, size_t N>
noinline T shortvec_bar(T<N> x, T<N> y) {
    T<N> soaValue;
    foreach (i = 0 ... N) {
        soaValue[i] = x[i] + y[i];
    }
    return soaValue[2];
}

int test_shortvec(uniform int c) {
    uniform int<3> x3 = {c, 2 * c, 3 * c};
    uniform int<3> y3 = {3 * c, 2 * c, c};
    return shortvec_bar<varying int, 3>(x3, y3);
}
