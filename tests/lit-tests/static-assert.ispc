// RUN: %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s

constexpr uniform int triple(uniform int x) {
    static_assert(sizeof(uniform int) == 4, "Expected a 32-bit integer.");
    return 3 * x;
}

static_assert(triple(3) == 9);

constexpr uniform int later();
constexpr uniform int limit = later();
static_assert(limit == 9, "Expected the deferred constexpr value.");

template <int N>
uniform int checked_limit() {
    static_assert(limit == N, "Expected the deferred global to match the template argument.");
    return N;
}

uniform int test_deferred_global() { return checked_limit<9>(); }

template <int N>
constexpr uniform int checked() {
    constexpr uniform int value = triple(N);
    static_assert(value, "The computed value must be nonzero.");
    return value;
}

constexpr uniform int later() { return checked<3>(); }

// CHECK-LABEL: @test_static_assert
// CHECK-NOT: call
// CHECK: ret i32 9
uniform int test_static_assert() {
    constexpr uniform int value = checked<3>();
    static_assert(value == 9, "Expected the instantiated value.");
    return value;
}
