// Check out of bounds warning for arrays inside templates.
// RUN: %{ispc} --target=host --nowrap --nostdlib %s 2>&1 | FileCheck %s

// CHECK-COUNT-1: Warning: Array index "8" may be out of bounds for 8 element array.

template <int C> void testArrayAccess(uniform int ret[], uniform int arr[])
{
    uniform int O[C] = {0};
    ret[0] = O[arr[0] % C]; // Valid access
    ret[1] = O[C]; // Non-Valid access
}

void test(uniform int ret[], uniform int arr[]) {
    testArrayAccess<8>(ret, arr);
}