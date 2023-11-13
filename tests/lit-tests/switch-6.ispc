// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: "continue" statement illegal outside of for/while/do/foreach loops

void foo(float f) {
    switch (f) {
    case 1:
        ++f;
        continue;
    case 2:
        f = 0;
    }
}

