// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

uniform int g = 1;
constexpr uniform int badinit = g;

constexpr uniform int noinit;

uniform int nonconstexpr(uniform int x) { return x + 1; }
constexpr uniform int badcall(uniform int x) { return nonconstexpr(x); }

constexpr uniform int badstmt(uniform int x) {
    print("x=%d", x);
    return x;
}

// CHECK: Error: Initializer for global variable "badinit" must be a constant.
// CHECK: Error: Missing initializer for constexpr variable "noinit".
// CHECK: Error: constexpr function "badcall" calls non-constexpr function "nonconstexpr".
// CHECK: Error: constexpr function "badstmt" contains disallowed statement "print".
