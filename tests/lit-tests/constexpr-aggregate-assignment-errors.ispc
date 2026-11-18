// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

constexpr uniform int write_past_end(uniform int i) {
    uniform int a[2] = {0};
    a[i] = 1;
    return a[0];
}

constexpr uniform int decrement_before_start(uniform int i) {
    uniform int<2> v = {1, 2};
    return v[i]--;
}

constexpr uniform int past_end = write_past_end(2);
constexpr uniform int before_start = decrement_before_start(-1);

// CHECK: Error: constexpr array/vector index is out of bounds.
// CHECK: Error: constexpr array/vector index is out of bounds.

constexpr varying int masked_write(varying bool cond) {
    varying int a[2] = {0, 1};
    varying int result = cond ? (a[0] = 3) : 0;
    return a[0];
}

constexpr varying int bad_mask = masked_write(programIndex == 0);

// CHECK: Error: Initializer for global variable "bad_mask" must be a constant.

constexpr varying int varying_index(varying int i) {
    varying int a[2] = {0, 1};
    a[i] = 3;
    return a[0];
}

constexpr varying int bad_index = varying_index(programIndex & 1);

// CHECK: Error: constexpr array/vector index must be a uniform constant.
