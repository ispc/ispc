// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Case statement value must be a compile-time integer constant

void foo1(float f) {
    switch (f) {
    case 1.5:
        ++f;
    }
}

// CHECK: Error: Duplicate case value "1"

void foo2(float f) {
    switch (f) {
    case 1:
        ++f;
    case 2:
    case 1:
        f = 0;
    }
}

// CHECK: Error: "case" label illegal outside of "switch" statement

void foo3(float f) {
    switch (f) {
    case 1:
        ++f;
    case 2:
        f = 0;
    }
 case 3:
     --f;
}

// CHECK: Error: "default" label illegal outside of "switch" statement

void foo4(float f) {
  default:
     ++f;
    switch (f) {
    case 1:
        ++f;
    case 2:
        f = 0;
    }
}

// CHECK: Error: "default" label illegal outside of "switch" statement

void foo5(float f) {
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

// CHECK: Error: "continue" statement illegal outside of for/while/do/foreach loops

void foo6(float f) {
    switch (f) {
    case 1:
        ++f;
        continue;
    case 2:
        f = 0;
    }
}