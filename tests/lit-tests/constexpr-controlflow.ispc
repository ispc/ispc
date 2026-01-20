// RUN: not %{ispc} %s --target=host --nostdlib -o /dev/null 2>&1 | FileCheck %s

constexpr uniform int bad(varying int x) {
    if (x > 0) {
        return 1;
    }
    return 2;
}

// CHECK: Error: constexpr function
// CHECK: "bad" requires a uniform condition in "if".
