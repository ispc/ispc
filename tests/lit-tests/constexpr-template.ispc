// RUN: %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s

constexpr uniform int width() { return 3; }
constexpr uniform int next(uniform int n) { return n + 1; }

template <int N>
uniform int foo() { return N; }

// CHECK-LABEL: @bar
// CHECK: ret i32 3
uniform int bar() { return foo<width()>(); }

template <int N>
uniform int add() { return foo<N + 2>(); }

template <int N>
uniform int call() { return foo<next(N)>(); }

template <int N>
uniform int local() {
    constexpr uniform int value = next(N);
    return foo<value>();
}

template <typename T>
uniform int size() { return foo<sizeof(uniform T)>(); }

// CHECK-LABEL: @test_add
// CHECK: ret i32 5
uniform int test_add() { return add<3>(); }

// CHECK-LABEL: @test_call
// CHECK: ret i32 7
uniform int test_call() { return call<6>(); }

// CHECK-LABEL: @test_local
// CHECK: ret i32 9
uniform int test_local() { return local<8>(); }

// CHECK-LABEL: @test_ordinary_local
// CHECK: ret i32 12
uniform int test_ordinary_local() {
    constexpr uniform int value = next(10);
    return foo<value + 1>();
}

// CHECK-LABEL: @test_size
// CHECK: ret i32 4
uniform int test_size() { return size<uniform int32>(); }

template <uint64 N>
uniform uint64 wide_leaf() { return N; }

template <uint64 N>
uniform uint64 wide() { return wide_leaf<N + 1>(); }

// CHECK-LABEL: @test_wide
// CHECK: ret i64 4294967297
uniform uint64 test_wide() { return wide<4294967296ull>(); }

struct Result {
    uniform int value;
};

template <int N>
uniform Result produce_result() {
    uniform Result result = {N};
    return result;
}

template <int N>
uniform int local_member() {
    constexpr uniform int value = next(N);
    return produce_result<value>().value;
}

// CHECK-LABEL: @test_local_member
// CHECK: ret i32 15
uniform int test_local_member() { return local_member<14>(); }

enum Number { TEN = 10 };
constexpr uniform Number identity(uniform Number n) { return n; }

template <Number N>
uniform int enum_leaf() { return N; }

template <Number N>
uniform int enum_argument() { return enum_leaf<identity(N)>(); }

// CHECK-LABEL: @test_enum
// CHECK: ret i32 10
uniform int test_enum() { return enum_argument<TEN>(); }

template <int N>
uniform int intrinsic_array_size() {
    uniform int arr[sizeof(@llvm.ctpop(N)) + sizeof(@llvm.ctpop(3ll))];
    return sizeof(arr);
}

// CHECK-LABEL: @test_intrinsic_array_size
// CHECK: ret i32 48
uniform int test_intrinsic_array_size() { return intrinsic_array_size<1>(); }
