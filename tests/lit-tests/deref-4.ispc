// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to dereference non-pointer type "varying float"

float func(float a) {
    *a = 0;
    return 0;
}
