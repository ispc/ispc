// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

constexpr varying int masked_increment(varying bool cond) {
    varying int count = 0;
    varying bool result = cond && (++count != 0);
    return count;
}

constexpr varying int masked_assignment(varying bool cond) {
    varying int count = 0;
    varying int result = cond ? (count = 1) : 0;
    return count;
}

// Compile-time evaluation cannot update individual lanes of a local variable.
constexpr varying int invalid_and = masked_increment(programIndex == 0);
constexpr varying int invalid_select = masked_assignment(programIndex == 0);

constexpr uniform int bad(varying int x) {
    if (x > 0) {
        return 1;
    }
    return 2;
}

// CHECK-DAG: Error: Initializer for global variable "invalid_and" must be a constant.
// CHECK-DAG: Error: Initializer for global variable "invalid_select" must be a constant.
// CHECK-DAG: Error: constexpr function "bad" requires a uniform condition in "if".
