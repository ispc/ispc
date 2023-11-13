// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: "signed" qualifier is illegal with non-integer type "varying float"

int foo() {
    signed float x;
}
