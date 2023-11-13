// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Duplicate case value "1"

void foo(float f) {
    switch (f) {
    case 1:
        ++f;
    case 2:
    case 1:
        f = 0;
    }
}

