// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: "default" label illegal outside of "switch" statement

void foo(float f) {
  default:
     ++f;
    switch (f) {
    case 1:
        ++f;
        continue;
    case 2:
        f = 0;
    }
}

