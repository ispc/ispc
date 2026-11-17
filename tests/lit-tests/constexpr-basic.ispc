// RUN: %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s

constexpr uniform int A = 7;
constexpr uniform int B = A * 2;

constexpr uniform int add(uniform int x, uniform int y) { return x + y; }

uniform int g[B];
uniform int h[add(3, 4)];

uniform int not_constexpr(uniform int x) { return x; }

constexpr uniform int sizeof_unevaluated() { return sizeof(not_constexpr(1)); }

uniform int s[sizeof_unevaluated()];

// CHECK: @g = {{.*}} [14 x i32]
// CHECK: @h = {{.*}} [7 x i32]
// CHECK: @s = {{.*}} [4 x i32]

constexpr uniform int count_and(uniform bool cond) {
    uniform int count = 0;
    uniform bool result = cond && (++count > 0);
    return count + (uniform int)result;
}

constexpr uniform int count_or(uniform bool cond) {
    uniform int count = 0;
    uniform bool result = cond || (++count > 0);
    return count + (uniform int)result;
}

constexpr uniform int count_select(uniform bool cond) {
    uniform int count = 0;
    uniform int result = cond ? (count += 2) : (count += 3);
    return count + result;
}

// The skipped operands must leave the local state untouched.
constexpr uniform int and_false = count_and(false);
constexpr uniform int and_true = count_and(true);
constexpr uniform int or_false = count_or(false);
constexpr uniform int or_true = count_or(true);
constexpr uniform int select_false = count_select(false);
constexpr uniform int select_true = count_select(true);

// CHECK-DAG: @and_false = {{.*}} constant i32 0{{(,|$)}}
// CHECK-DAG: @and_true = {{.*}} constant i32 2{{(,|$)}}
// CHECK-DAG: @or_false = {{.*}} constant i32 2{{(,|$)}}
// CHECK-DAG: @or_true = {{.*}} constant i32 1{{(,|$)}}
// CHECK-DAG: @select_false = {{.*}} constant i32 6{{(,|$)}}
// CHECK-DAG: @select_true = {{.*}} constant i32 4{{(,|$)}}
