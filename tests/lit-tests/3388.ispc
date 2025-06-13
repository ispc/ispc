// This test checks that the compiler reports an error when performing an 
// extra index access in aggregate initialization instead of a crash 
// (segmentation fault).

// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Trying to index into non-array, vector, or pointer type "const uniform float".
// CHECK-NOT: FATAL ERROR:

export void test(const uniform float a[]) {
    const uniform float v = {a[0][0]};
}
