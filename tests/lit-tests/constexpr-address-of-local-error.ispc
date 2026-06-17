// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

constexpr uniform int bad_addr() {
    uniform int x = 1;
    uniform int *uniform p = &x;
    return p != NULL;
}

uniform int arr[bad_addr()];

// CHECK: Error: constexpr function "bad_addr" contains disallowed expression "address-of".
