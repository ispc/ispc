// REQUIRES: X86_ENABLED && X86_64_HOST
// RUN: not %{ispc} %s --target=generic-i32x4 --nostdlib -o /dev/null 2>&1 | FileCheck %s

struct Pair {
    uniform int a;
    uniform int b;
};

constexpr uniform int bad_arr[2] = {1, 2, 3};
constexpr uniform Pair bad_struct = {1, 2, 3};

constexpr uniform int<2> v2 = {1, 2};
constexpr uniform int<2> bad_swizzle_char = v2.qy;
constexpr uniform int<2> bad_swizzle_oob = v2.zw;

// CHECK: Error:
// CHECK: Initializer list
// CHECK: for array
// CHECK: "const uniform int32[2]"
// CHECK: must have
// CHECK: no more
// CHECK: than 2
// CHECK: elements
// CHECK: (has 3).
// CHECK: Error:
// CHECK: Initializer list
// CHECK: for struct
// CHECK: "const uniform struct Pair"
// CHECK: must
// CHECK: have no
// CHECK: more than 2
// CHECK: elements
// CHECK: (has 3).
// CHECK: identifier "qy" unknown.
// CHECK: character 'z' for constexpr vector.
