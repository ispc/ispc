// REQUIRES: X86_ENABLED && X86_64_HOST
// RUN: not %{ispc} %s --target=generic-i32x4 --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

struct Pair {
    uniform int a;
    uniform int b;
};

constexpr uniform int bad_arr[2] = {1, 2, 3};
constexpr uniform Pair bad_struct = {1, 2, 3};

constexpr uniform int<2> v2 = {1, 2};
constexpr uniform int<2> bad_swizzle_char = v2.qy;
constexpr uniform int<2> bad_swizzle_oob = v2.zw;

// CHECK: Error: Initializer list for array "const uniform int32[2]" must have no more than 2 elements (has 3).
// CHECK: Error: Initializer list for struct "const uniform struct Pair" must have no more than 2 elements (has 3).
// CHECK: Error: Vector element identifier "qy" unknown.
// CHECK: Error: Invalid swizzle character 'z' for constexpr vector.
