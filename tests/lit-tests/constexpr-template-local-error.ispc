// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

template <int N>
uniform int initialize(uniform int runtime) {
    constexpr uniform int local = runtime;
    return N + local;
}

uniform int test(uniform int runtime) { return initialize<1>(runtime); }

// CHECK: Error: Initializer for constexpr variable "local" must be a compile-time constant.
