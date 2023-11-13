// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't assign to type "const uniform int32 * const varying"

void foo(const int * const p) {
    ++p;
}
