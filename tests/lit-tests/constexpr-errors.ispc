// RUN: not %{ispc} %s --target=host --nostdlib -o /dev/null 2>&1 | FileCheck %s

uniform int g = 1;
constexpr uniform int badinit = g;

constexpr uniform int noinit;

uniform int nonconstexpr(uniform int x) { return x + 1; }
constexpr uniform int badcall(uniform int x) { return nonconstexpr(x); }

constexpr uniform int badstmt(uniform int x) {
    print("x=%d", x);
    return x;
}

// CHECK: Error:
// CHECK: Initializer for
// CHECK: global
// CHECK: variable "badinit" must be a constant.
// CHECK: Error:
// CHECK: Missing initializer for
// CHECK: constexpr variable "noinit".
// CHECK: Error:
// CHECK: constexpr function
// CHECK: "badcall" calls non-constexpr function "nonconstexpr".
// CHECK: Error:
// CHECK: constexpr function
// CHECK: "badstmt" contains disallowed statement "print".
