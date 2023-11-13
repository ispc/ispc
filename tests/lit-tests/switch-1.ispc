// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Case statement value must be a compile-time integer constant

void foo(float f) {
    switch (f) {
    case 1.5:
        ++f;
    }
}

