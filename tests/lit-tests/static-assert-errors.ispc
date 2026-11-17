// RUN: not %{ispc} %s --target=host --nostdlib --nowrap --opt=disable-assertions -o /dev/null 2>&1 | FileCheck %s
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -DDEFERRED -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFERRED

#ifdef DEFERRED
constexpr uniform int later();
// DEFERRED: Error: static assertion failed: Deferred assertion.
static_assert(later() > 0, "Deferred assertion.");
constexpr uniform int later() { return 0; }
#else

// CHECK: Error: static assertion failed: File scope assertion.
static_assert(false, "File scope assertion.");

// CHECK: Error: static assertion failed.
static_assert(false);

void invalid_conditions(uniform int runtime, varying int lanes) {
    // CHECK: Error: static_assert condition must be a compile-time constant.
    static_assert(runtime);
    // CHECK: Error: static_assert condition must have a uniform scalar type convertible to bool.
    static_assert(lanes != 0);
    constexpr uniform int<2> values = {1, 2};
    // CHECK: Error: static_assert condition must have a uniform scalar type convertible to bool.
    static_assert(values);
    // CHECK: Error: static assertion failed: Assertion in a dead branch.
    if (false) {
        static_assert(false, "Assertion in a dead branch.");
    }
}

constexpr uniform int invalid_parameter(uniform const int x) {
    // CHECK: Error: static_assert condition must be a compile-time constant.
    static_assert(x);
    return x;
}

template <int N>
uniform int positive() {
    // CHECK: Error: static assertion failed: The template argument must be positive.
    static_assert(N > 0, "The template argument must be positive.");
    return N;
}

uniform int test_invalid_instantiation() { return positive<0>(); }
#endif
